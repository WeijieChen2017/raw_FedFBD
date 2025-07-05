import torch
import torch.nn as nn
from monai.networks.nets import UNet


class FBDUNet(nn.Module):
    """
    UNet model for SIIM segmentation that can be split into FBD parts.
    Based on MONAI UNet with 5 levels.
    """
    def __init__(self, in_channels=1, out_channels=1, features=128):
        super(FBDUNet, self).__init__()
        
        # Create the full UNet model
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(features, features*2, features*4, features*8, features*16),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            dropout=0.2
        )
        
        # Define FBD model parts based on UNet structure
        # UNet has encoder, bottleneck, and decoder parts
        # We'll split it into logical parts for FBD
        
        # Part 1: Initial convolution and first encoder level
        self.encoder_level1 = nn.Sequential(
            self.unet.model[0],  # Initial convolution
            self.unet.model[1][0]  # First encoder block
        )
        
        # Part 2: Second encoder level
        self.encoder_level2 = self.unet.model[1][1]
        
        # Part 3: Third encoder level
        self.encoder_level3 = self.unet.model[1][2]
        
        # Part 4: Fourth encoder level
        self.encoder_level4 = self.unet.model[1][3]
        
        # Part 5: Bottleneck (deepest level)
        self.bottleneck = self.unet.model[1][4]
        
        # Part 6: First decoder level
        self.decoder_level1 = self.unet.model[2][0]
        
        # Part 7: Second decoder level
        self.decoder_level2 = self.unet.model[2][1]
        
        # Part 8: Third decoder level
        self.decoder_level3 = self.unet.model[2][2]
        
        # Part 9: Fourth decoder level and output
        self.decoder_level4 = nn.Sequential(
            self.unet.model[2][3],
            self.unet.model[2][4]  # Final output convolution
        )
        
    def forward(self, x):
        # For standard forward pass, use the complete UNet
        return self.unet(x)
    
    def get_fbd_parts(self):
        """
        Return the model parts for FBD framework.
        """
        return {
            'encoder_level1': self.encoder_level1,
            'encoder_level2': self.encoder_level2,
            'encoder_level3': self.encoder_level3,
            'encoder_level4': self.encoder_level4,
            'bottleneck': self.bottleneck,
            'decoder_level1': self.decoder_level1,
            'decoder_level2': self.decoder_level2,
            'decoder_level3': self.decoder_level3,
            'decoder_level4': self.decoder_level4
        }


def get_siim_model(architecture="unet", in_channels=1, out_channels=1, features=128):
    """
    Factory function to create SIIM models.
    """
    if architecture.lower() == "unet":
        return FBDUNet(in_channels=in_channels, out_channels=out_channels, features=features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


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