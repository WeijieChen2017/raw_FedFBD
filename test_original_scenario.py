#!/usr/bin/env python3
"""
Test the original scenario that was causing the TypeError
"""

import torch
from fbd_models_siim import get_pretrained_fbd_model

print("Testing original scenario that caused the TypeError...")

try:
    # This was the original call that was failing
    model = get_pretrained_fbd_model(
        architecture="unet", 
        norm=None, 
        in_channels=1, 
        num_classes=1, 
        use_pretrained=False
    )
    
    print("✓ Model created successfully!")
    print(f"Model type: {type(model)}")
    
    # Test getting FBD parts (this was where the error occurred)
    parts = model.get_fbd_parts()
    print(f"✓ FBD parts retrieved successfully! Got {len(parts)} parts:")
    
    for name in parts.keys():
        print(f"  - {name}")
    
    # Test a forward pass
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    
    print("\n🎉 All tests passed! The TypeError has been fixed.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()