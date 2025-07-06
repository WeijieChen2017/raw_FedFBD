#!/usr/bin/env python3
"""
Debug script to map model parts to their parameters
"""

import torch
from fbd_models_siim import get_pretrained_fbd_model

print("Creating SIIM UNet model...")
model = get_pretrained_fbd_model(
    architecture="unet",
    norm=None,
    in_channels=1,
    num_classes=1,
    use_pretrained=False
)

print("\n=== Full Model Parameter Names ===")
full_param_names = dict(model.named_parameters())
print(f"Total model parameters: {len(full_param_names)}")

print("\n=== FBD Parts Parameter Names ===")
parts = model.get_fbd_parts()

for part_name, part_module in parts.items():
    part_params = dict(part_module.named_parameters())
    print(f"\n{part_name}: {len(part_params)} parameters")
    
    # Find the corresponding parameters in the full model
    full_model_param_names = []
    for full_name, full_param in full_param_names.items():
        for part_param_name, part_param in part_params.items():
            # Check if this is the same parameter object
            if full_param is part_param:
                full_model_param_names.append(full_name)
    
    print(f"  Corresponding full model parameters:")
    for name in full_model_param_names[:5]:  # Show first 5
        print(f"    {name}")
    if len(full_model_param_names) > 5:
        print(f"    ... and {len(full_model_param_names) - 5} more")