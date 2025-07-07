import torch
import torch.nn as nn
from monai.networks.nets import UNet


class FBDUNet(nn.Module):
    """
    UNet model for SIIM segmentation that can be split into FBD parts.
    Based on MONAI UNet with 3 levels.
    """
    def __init__(self, in_channels, out_channels, features):
        super(FBDUNet, self).__init__()
        
        # Determine UNet channel configuration based on 'features'
        # For SIIM data with roi_size [128, 128, 32], use different strides for Z dimension
        if features == 128:
            channels = (128, 256, 512)
            strides = ((2, 2, 1), (2, 2, 2))  # Less aggressive downsampling in Z
        elif features == 64:
            channels = (64, 128, 256)
            strides = ((2, 2, 1), (2, 2, 2))  # Less aggressive downsampling in Z
        elif features == 32:
            channels = (32, 64, 128)
            strides = ((2, 2, 1), (2, 2, 2))  # Less aggressive downsampling in Z
        else:
            # Default to a reasonable configuration if features is not one of the presets
            channels = (features, features * 2, features * 4)
            strides = ((2, 2, 1), (2, 2, 2))  # Less aggressive downsampling in Z

        self.unet = UNet(
            spatial_dims=3,  # Changed to 3D for SIIM volume data
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2
        )
        
    def forward(self, x):
        # For standard forward pass, use the complete UNet
        return self.unet(x)
    
    def get_fbd_parts(self):
        """
        Return the model parts for FBD framework.
        For SIIM UNet, we treat the entire model as a single part to avoid
        complex decomposition of the nested MONAI UNet structure.
        """
        # Return the entire model as a single part
        return {
            'unet': self.unet
        }


def get_siim_model(architecture="unet", in_channels=1, out_channels=1, model_size="standard"):
    """
    Returns a U-Net model for the SIIM dataset.
    
    Args:
        architecture (str): The model architecture to use (only "unet" supported).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        model_size (str): 'standard' or 'small' to select feature set.
        
    Returns:
        A PyTorch model.
    """
    if architecture != "unet":
        raise ValueError(f"Unsupported architecture for SIIM: {architecture}")
    
    if model_size == 'small':
        # Use smaller feature size for reduced memory usage
        features = 128
    elif model_size == 'standard':
        # Default feature size
        features = 128
    else:
        raise ValueError(f"Unsupported model_size: {model_size}. Choose 'standard' or 'small'.")
    
    # Return FBDUNet which has the necessary FBD methods
    model = FBDUNet(in_channels=in_channels, out_channels=out_channels, features=features)
    return model


# For compatibility with FBD framework
def get_pretrained_fbd_model(architecture="unet", norm=None, in_channels=1, num_classes=1, use_pretrained=False):
    """
    Interface function compatible with FBD framework.
    For SIIM, we don't use pretrained models.
    """
    if architecture.lower() != "unet":
        raise ValueError(f"SIIM only supports UNet architecture, got {architecture}")
    
    # For segmentation, num_classes is the number of output channels
    model = FBDUNet(in_channels=in_channels, out_channels=num_classes, features=128)
    
    if use_pretrained:
        print("Warning: No pretrained weights available for SIIM UNet model")
    
    return model