#!/usr/bin/env python3
"""
Test script to instantiate UNet model and generate weights state_dict
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
import traceback

# Test 1: Create MONAI UNet directly
print("=== Test 1: Direct MONAI UNet ===")
unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
print(f"UNet created successfully")
print(f"Type of model: {type(unet.model)}")
print(f"Model structure: {unet.model}")

# Test 2: Access model parts
print("\n=== Test 2: Accessing model parts ===")
try:
    print(f"model[0] (initial conv): {type(unet.model[0])}")
    print(f"model[1] (encoder): {type(unet.model[1])}")
    print(f"model[2] (decoder): {type(unet.model[2])}")
    
    # Check if model[1] is subscriptable
    print(f"\nEncoder type check:")
    print(f"  Is ModuleList: {isinstance(unet.model[1], nn.ModuleList)}")
    print(f"  Is Sequential: {isinstance(unet.model[1], nn.Sequential)}")
    print(f"  Has __getitem__: {hasattr(unet.model[1], '__getitem__')}")
    
    if hasattr(unet.model[1], '__len__'):
        print(f"  Length: {len(unet.model[1])}")
        print(f"\nEncoder blocks:")
        for i in range(len(unet.model[1])):
            print(f"  model[1][{i}]: {type(unet.model[1][i])}")
    else:
        print("  model[1] is not indexable")
        print(f"  Attributes: {[attr for attr in dir(unet.model[1]) if not attr.startswith('_')]}")
except Exception as e:
    print(f"Error accessing model parts: {e}")
    traceback.print_exc()

# Test 3: Generate state dict
print("\n=== Test 3: State dict keys ===")
state_dict = unet.state_dict()
print(f"Total parameters: {len(state_dict)}")
print("Sample keys:")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {key}: {state_dict[key].shape}")

# Test 4: Analyze SkipConnection structure
print("\n=== Test 4: Analyzing SkipConnection structure ===")
skip_conn = unet.model[1]
print(f"SkipConnection type: {type(skip_conn)}")
print(f"SkipConnection has submodule: {hasattr(skip_conn, 'submodule')}")
if hasattr(skip_conn, 'submodule'):
    print(f"Submodule type: {type(skip_conn.submodule)}")
    print(f"Submodule length: {len(skip_conn.submodule) if hasattr(skip_conn.submodule, '__len__') else 'N/A'}")
    
    if isinstance(skip_conn.submodule, nn.Sequential):
        print("\nSubmodule contents:")
        for i, module in enumerate(skip_conn.submodule):
            print(f"  [{i}] {type(module).__name__}")
            if hasattr(module, 'submodule'):
                print(f"      Has submodule: {type(module.submodule).__name__}")

# Test 5: Create a simple UNet that mimics MONAI structure
print("\n=== Test 5: Creating simplified FBD UNet ===")
class FBDUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=128):
        super().__init__()
        channels = (features, features*2, features*4, features*8, features*16)
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        
        # The MONAI UNet structure is nested, we need to handle it properly
        self.initial_conv = self.unet.model[0]
        self.skip_connection = self.unet.model[1]
        self.final_layers = self.unet.model[2]
        
    def forward(self, x):
        return self.unet(x)
    
    def get_fbd_parts(self):
        # For now, return the main components
        # We'll need to further decompose these based on the actual structure
        return {
            "initial_conv": self.initial_conv,
            "encoder_decoder": self.skip_connection,
            "final_layers": self.final_layers,
        }

# Test FBDUNet
try:
    fbd_unet = FBDUNet(in_channels=1, out_channels=1, features=128)
    print("FBDUNet created successfully")
    
    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)
    output = fbd_unet(x)
    print(f"Forward pass successful: input {x.shape} -> output {output.shape}")
    
    # Get FBD parts
    parts = fbd_unet.get_fbd_parts()
    print(f"\nFBD parts:")
    for name, part in parts.items():
        print(f"  {name}: {type(part)}")
        
except Exception as e:
    print(f"Error creating FBDUNet: {e}")
    import traceback
    traceback.print_exc()