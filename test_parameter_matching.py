#!/usr/bin/env python3
"""
Test the parameter matching logic for SIIM FBD training
"""

import torch
from fbd_models_siim import get_pretrained_fbd_model

print("Testing parameter matching logic...")

# Create model
model = get_pretrained_fbd_model(
    architecture="unet",
    norm=None,
    in_channels=1,
    num_classes=1,
    use_pretrained=False
)

print(f"Total model parameters: {len(list(model.parameters()))}")

# Test the parameter mapping logic (same as in fixed client code)
fbd_parts = model.get_fbd_parts()
component_to_param_names = {}

for component_name, component_module in fbd_parts.items():
    component_to_param_names[component_name] = []
    # Get all parameters from this component
    component_params = dict(component_module.named_parameters())
    
    # Find corresponding parameter names in the full model
    for full_name, full_param in model.named_parameters():
        for comp_param_name, comp_param in component_params.items():
            if full_param is comp_param:
                component_to_param_names[component_name].append(full_name)

print("\n=== Component to Parameter Mapping ===")
total_mapped_params = 0
for component_name, param_names in component_to_param_names.items():
    print(f"{component_name}: {len(param_names)} parameters")
    total_mapped_params += len(param_names)

print(f"\nTotal mapped parameters: {total_mapped_params}")
print(f"Total model parameters: {len(list(model.parameters()))}")

if total_mapped_params == len(list(model.parameters())):
    print("✅ All parameters correctly mapped!")
else:
    print("❌ Parameter mapping incomplete!")

# Test parameter freezing/unfreezing
print("\n=== Testing Parameter Freezing ===")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

trainable_count_before = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters after freezing: {trainable_count_before}")

# Unfreeze parameters for one component (e.g., initial_conv)
test_component = 'initial_conv'
if test_component in component_to_param_names:
    for param_name in component_to_param_names[test_component]:
        for name, param in model.named_parameters():
            if name == param_name:
                param.requires_grad = True

trainable_count_after = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters after unfreezing {test_component}: {trainable_count_after}")

expected_params = len(component_to_param_names[test_component])
if trainable_count_after == expected_params:
    print(f"✅ Successfully unfroze {expected_params} parameters for {test_component}")
else:
    print(f"❌ Expected {expected_params} trainable parameters, got {trainable_count_after}")

# Test optimizer creation
print("\n=== Testing Optimizer Creation ===")
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters for optimizer: {len(trainable_params)}")

if len(trainable_params) > 0:
    try:
        import torch.optim as optim
        optimizer = optim.Adam(trainable_params, lr=0.001)
        print("✅ Optimizer created successfully!")
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
else:
    print("❌ No trainable parameters found for optimizer!")