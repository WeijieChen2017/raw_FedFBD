#!/usr/bin/env python3
"""
Generate UNet weights state_dict for SIIM FBD implementation
"""

import torch
from fbd_models_siim import FBDUNet

print("Generating UNet weights state_dict...")

# Create a UNet model with the SIIM configuration
model = FBDUNet(in_channels=1, out_channels=1, features=128)

# Get the state dict
state_dict = model.state_dict()

# Save the weights
torch.save(state_dict, 'siim_unet_weights.pth')

print(f"✓ Saved state dict with {len(state_dict)} parameters to 'siim_unet_weights.pth'")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Also save FBD parts information
parts = model.get_fbd_parts()
parts_info = {}
for name, part in parts.items():
    param_count = sum(p.numel() for p in part.parameters())
    parts_info[name] = {
        'type': type(part).__name__,
        'parameters': param_count
    }

import json
with open('siim_unet_parts_info.json', 'w') as f:
    json.dump(parts_info, f, indent=2)

print("✓ Saved FBD parts information to 'siim_unet_parts_info.json'")
print("\nModel is ready for use!")