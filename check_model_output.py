#!/usr/bin/env python3
"""
Check what the SIIM UNet model outputs (raw logits, sigmoid, softmax?)
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fbd_models_siim import get_siim_model

def check_model_output():
    print("🔍 Checking SIIM UNet Model Output")
    print("=" * 50)
    
    # Create model
    model = get_siim_model(
        architecture="unet",
        in_channels=1,
        out_channels=1,  # Binary segmentation
        model_size="small"
    )
    
    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 128, 32)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Get model output
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\nModel Output Analysis:")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output mean: {output.mean():.3f}")
    print(f"Output std: {output.std():.3f}")
    
    # Check if output looks like:
    # 1. Raw logits (can be any range)
    # 2. Sigmoid output (0 to 1)
    # 3. Softmax output (0 to 1, but different distribution)
    
    if output.min() >= 0 and output.max() <= 1:
        if torch.allclose(output.sum(dim=1), torch.ones(output.shape[0])):
            print("🎯 Output looks like: SOFTMAX probabilities")
        else:
            print("🎯 Output looks like: SIGMOID probabilities")
    else:
        print("🎯 Output looks like: RAW LOGITS (no activation)")
    
    # Test with different activations
    print(f"\nTesting Different Activations:")
    sigmoid_output = torch.sigmoid(output)
    print(f"After sigmoid: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
    
    # For binary segmentation, we don't need softmax since we only have 1 output channel
    # But let's check what happens if we add a background channel
    if output.shape[1] == 1:
        print("✅ Model has 1 output channel - correct for binary segmentation with sigmoid")
        print("❌ Should NOT use softmax for binary segmentation")
    else:
        print(f"⚠️  Model has {output.shape[1]} output channels")
        softmax_output = torch.softmax(output, dim=1)
        print(f"After softmax: [{softmax_output.min():.3f}, {softmax_output.max():.3f}]")
    
    return output

def check_monai_unet_default():
    """Check what MONAI UNet does by default"""
    print(f"\n🔬 MONAI UNet Default Behavior:")
    print("-" * 40)
    
    from monai.networks.nets import UNet
    
    # Create basic MONAI UNet
    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,  # Binary segmentation
        channels=(32, 64, 128),
        strides=(2, 2),
        num_res_units=2
    )
    
    # Test input
    test_input = torch.randn(1, 1, 64, 64, 16)
    
    unet.eval()
    with torch.no_grad():
        output = unet(test_input)
    
    print(f"MONAI UNet output range: [{output.min():.3f}, {output.max():.3f}]")
    
    if output.min() >= 0 and output.max() <= 1:
        print("✅ MONAI UNet outputs probabilities (has activation)")
    else:
        print("✅ MONAI UNet outputs raw logits (no activation)")
        print("   This is correct - loss function should handle activation")

def recommend_fix():
    print(f"\n💡 Recommendations:")
    print("=" * 30)
    
    print("For binary segmentation with 1 output channel:")
    print("✅ CORRECT: Model outputs raw logits")
    print("✅ CORRECT: Use sigmoid activation in loss function")
    print("❌ WRONG: Don't use softmax for binary segmentation")
    print("❌ WRONG: Don't apply sigmoid twice")
    
    print(f"\nProper SIIM Setup:")
    print("1. Model: UNet with out_channels=1 (binary segmentation)")
    print("2. Model output: Raw logits (no activation layer)")
    print("3. Loss function: DiceCELoss(sigmoid=True) or BCEWithLogitsLoss()")
    print("4. During inference: Apply sigmoid to get probabilities")
    
    print(f"\nIf loss is still going to zero, try:")
    print("• DiceCELoss(sigmoid=False) + manually apply sigmoid")
    print("• Pure BCE: BCEWithLogitsLoss()")
    print("• Check if targets are correct binary (0/1)")

if __name__ == "__main__":
    output = check_model_output()
    check_monai_unet_default() 
    recommend_fix()