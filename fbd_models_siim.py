import torch
import torch.nn as nn
from monai.networks.nets import UNet


class FBDUNet(nn.Module):
    """
    UNet model for SIIM segmentation that can be split into FBD parts.
    Based on MONAI UNet with 5 levels.
    """
    def __init__(self, in_channels, out_channels, features):
        super(FBDUNet, self).__init__()
        
        # Determine UNet channel configuration based on 'features'
        if features == 128:
            channels = (128, 256, 512)
            strides = (2, 2)
        elif features == 64:
            channels = (64, 128, 256)
            strides = (2, 2)
        elif features == 32:
            channels = (32, 64, 128)
            strides = (2, 2)
        else:
            # Default to a reasonable configuration if features is not one of the presets
            channels = (features, features * 2, features * 4)
            strides = (2, 2)

        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2
        )
        
        # Define FBD model parts based on actual MONAI UNet structure
        # MONAI UNet structure:
        # model[0]: Initial ResidualUnit
        # model[1]: SkipConnection with nested structure (encoder-decoder)
        # model[2]: Final Sequential layers
        
        # Part 1: Initial convolution
        self.initial_conv = self.unet.model[0]
        
        # The main encoder-decoder part is nested in skip connections
        # We need to access the submodules correctly
        main_skip = self.unet.model[1]  # Main SkipConnection
        
        # Part 2: First encoder level
        self.encoder_level1 = main_skip.submodule[0]  # First ResidualUnit
        
        # Part 3: Second level SkipConnection
        level2_skip = main_skip.submodule[1]  # SkipConnection
        self.encoder_level2 = level2_skip.submodule[0]  # Second ResidualUnit
        
        # Part 4: Third level SkipConnection
        level3_skip = level2_skip.submodule[1]  # SkipConnection
        self.encoder_level3 = level3_skip.submodule[0]  # Third ResidualUnit
        
        # Part 5: Fourth level SkipConnection (bottleneck)
        level4_skip = level3_skip.submodule[1]  # SkipConnection
        self.bottleneck = level4_skip.submodule  # Bottom ResidualUnit
        
        # Part 6: Decoder levels (upsampling blocks)
        self.decoder_level3 = level3_skip.submodule[2]  # First decoder block
        self.decoder_level2 = level2_skip.submodule[2]  # Second decoder block
        self.decoder_level1 = main_skip.submodule[2]    # Third decoder block
        
        # Part 7: Final output layers
        self.final_layers = self.unet.model[2]
        
    def forward(self, x):
        # For standard forward pass, use the complete UNet
        return self.unet(x)
    
    def get_fbd_parts(self):
        """
        Return the model parts for FBD framework.
        These parts correspond to different levels of the U-Net architecture.
        """
        return {
            'initial_conv': self.initial_conv,
            'encoder_level1': self.encoder_level1,
            'encoder_level2': self.encoder_level2,
            'encoder_level3': self.encoder_level3,
            'bottleneck': self.bottleneck,
            'decoder_level3': self.decoder_level3,
            'decoder_level2': self.decoder_level2,
            'decoder_level1': self.decoder_level1,
            'final_layers': self.final_layers
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
        features = 64
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