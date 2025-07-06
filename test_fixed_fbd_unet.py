#!/usr/bin/env python3
"""
Test script to verify the fixed FBD UNet implementation
"""

import torch
import torch.nn as nn
from fbd_models_siim import FBDUNet, get_siim_model, get_pretrained_fbd_model

print("=" * 80)
print("Testing Fixed FBD UNet Implementation")
print("=" * 80)

# Test 1: Create FBDUNet directly
print("\n=== Test 1: Direct FBDUNet instantiation ===")
try:
    model = FBDUNet(in_channels=1, out_channels=1, features=64)
    print("✓ FBDUNet created successfully")
    
    # Test forward pass
    x = torch.randn(1, 1, 32, 32, 32)  # Smaller size for faster testing
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    print(f"✓ Forward pass successful: {output.shape}")
    
except Exception as e:
    print(f"✗ Error creating FBDUNet: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Get FBD parts
print("\n=== Test 2: FBD parts access ===")
try:
    parts = model.get_fbd_parts()
    print(f"✓ FBD parts retrieved successfully")
    print(f"Number of parts: {len(parts)}")
    
    print("\nPart details:")
    for name, part in parts.items():
        print(f"  {name}: {type(part).__name__}")
        # Count parameters
        param_count = sum(p.numel() for p in part.parameters())
        print(f"    Parameters: {param_count:,}")
    
except Exception as e:
    print(f"✗ Error getting FBD parts: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Factory functions
print("\n=== Test 3: Factory functions ===")
try:
    model2 = get_siim_model(architecture="unet", in_channels=1, out_channels=2, features=32)
    print("✓ get_siim_model() works")
    
    model3 = get_pretrained_fbd_model(architecture="unet", in_channels=1, num_classes=3)
    print("✓ get_pretrained_fbd_model() works")
    
except Exception as e:
    print(f"✗ Error with factory functions: {e}")
    import traceback
    traceback.print_exc()

# Test 4: State dict compatibility
print("\n=== Test 4: State dict operations ===")
try:
    state_dict = model.state_dict()
    print(f"✓ State dict extracted: {len(state_dict)} parameters")
    
    # Test saving and loading
    torch.save(state_dict, 'test_unet_weights.pth')
    loaded_state = torch.load('test_unet_weights.pth')
    
    # Create new model and load weights
    new_model = FBDUNet(in_channels=1, out_channels=1, features=64)
    new_model.load_state_dict(loaded_state)
    print("✓ State dict save/load successful")
    
    # Clean up
    import os
    os.remove('test_unet_weights.pth')
    
except Exception as e:
    print(f"✗ Error with state dict operations: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Parameter counting
print("\n=== Test 5: Model statistics ===")
try:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
    
except Exception as e:
    print(f"✗ Error counting parameters: {e}")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)