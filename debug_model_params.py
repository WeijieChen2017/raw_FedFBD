#!/usr/bin/env python3
"""
Debug script to understand the model parameter structure
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

print("\n=== Model Structure ===")
for name, module in model.named_children():
    print(f"{name}: {type(module).__name__}")

print("\n=== FBD Parts ===")
parts = model.get_fbd_parts()
for name, part in parts.items():
    param_count = sum(p.numel() for p in part.parameters())
    print(f"{name}: {type(part).__name__} ({param_count:,} params)")

print("\n=== All Parameter Names (first 20) ===")
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 20:
        print(f"{name}: {param.shape}")
    else:
        break

print(f"\nTotal parameters: {len(list(model.named_parameters()))}")

print("\n=== Parameter Name Prefixes ===")
param_names = [name for name, _ in model.named_parameters()]
prefixes = set()
for name in param_names:
    # Get the top-level prefix (before the first dot)
    if '.' in name:
        prefix = name.split('.')[0]
        prefixes.add(prefix)

print(f"Top-level prefixes: {sorted(prefixes)}")

print("\n=== Testing Parameter Name Matching ===")
component_names = list(parts.keys())
print(f"Component names to match: {component_names}")

for component_name in component_names:
    matching_params = [name for name, _ in model.named_parameters() 
                      if name.startswith(component_name)]
    print(f"\n{component_name}: {len(matching_params)} matching parameters")
    if matching_params:
        print(f"  Examples: {matching_params[:3]}")
    else:
        print("  No matches found!")