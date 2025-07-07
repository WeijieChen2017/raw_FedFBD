#!/usr/bin/env python3
"""
Script to debug SIIM loss function issues.
"""

import torch
import numpy as np
from monai.losses import DiceCELoss
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dice_ce_loss():
    """Test DiceCELoss with different configurations to identify the issue."""
    
    print("🔍 Debugging SIIM DiceCELoss Configuration")
    print("=" * 60)
    
    # Create sample data similar to SIIM
    batch_size = 2
    height, width, depth = 128, 128, 32
    
    # Create sample predictions (model output)
    # Simulate output from UNet (raw logits)
    predictions = torch.randn(batch_size, 1, height, width, depth)
    
    # Create sample targets (ground truth masks)
    # Binary segmentation: 0 = background, 1 = pneumothorax
    targets = torch.zeros(batch_size, 1, height, width, depth)
    # Add some positive regions (simulate pneumothorax)
    targets[0, 0, 40:60, 40:60, 10:20] = 1.0  # Small region in first sample
    targets[1, 0, 30:80, 30:80, 5:15] = 1.0   # Larger region in second sample
    
    print(f"Input shapes:")
    print(f"  Predictions: {predictions.shape} (range: {predictions.min():.3f} to {predictions.max():.3f})")
    print(f"  Targets: {targets.shape} (unique values: {torch.unique(targets)})")
    print(f"  Positive voxels: {(targets == 1).sum()} / {targets.numel()} ({(targets == 1).sum() / targets.numel() * 100:.2f}%)")
    
    # Test different DiceCELoss configurations
    configs = [
        {"name": "Current Config", "to_onehot_y": False, "sigmoid": True},
        {"name": "Alternative 1", "to_onehot_y": False, "sigmoid": False},
        {"name": "Alternative 2", "to_onehot_y": True, "sigmoid": True},
        {"name": "Alternative 3", "to_onehot_y": True, "sigmoid": False},
    ]
    
    print(f"\n📊 Testing Different DiceCELoss Configurations:")
    print("-" * 60)
    
    for config in configs:
        try:
            print(f"\n{config['name']}:")
            print(f"  DiceCELoss(to_onehot_y={config['to_onehot_y']}, sigmoid={config['sigmoid']})")
            
            # Create loss function
            criterion = DiceCELoss(to_onehot_y=config['to_onehot_y'], sigmoid=config['sigmoid'])
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            print(f"  Loss value: {loss.item():.6f}")
            
            # Check if loss makes sense
            if loss.item() < 0.001:
                print("  ⚠️  WARNING: Loss is very small!")
            elif loss.item() > 10:
                print("  ⚠️  WARNING: Loss is very large!")
            else:
                print("  ✅ Loss seems reasonable")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    # Test with extreme cases
    print(f"\n🧪 Testing Edge Cases:")
    print("-" * 30)
    
    # Case 1: All background (no pneumothorax)
    targets_all_bg = torch.zeros_like(targets)
    
    # Case 2: All foreground (unrealistic but tests bounds)
    targets_all_fg = torch.ones_like(targets)
    
    # Case 3: Perfect prediction
    perfect_pred = targets.clone()
    
    edge_cases = [
        ("All background targets", predictions, targets_all_bg),
        ("All foreground targets", predictions, targets_all_fg),
        ("Perfect prediction", perfect_pred, targets),
    ]
    
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True)  # Current config
    
    for case_name, pred, tgt in edge_cases:
        try:
            loss = criterion(pred, tgt)
            print(f"{case_name}: {loss.item():.6f}")
        except Exception as e:
            print(f"{case_name}: ERROR - {e}")

def analyze_siim_data_properties():
    """Analyze properties of SIIM data that might affect loss calculation."""
    
    print(f"\n🔬 SIIM Data Analysis:")
    print("-" * 30)
    
    print("Expected SIIM characteristics:")
    print("• Task: Binary segmentation (pneumothorax detection)")
    print("• Classes: 0 = background, 1 = pneumothorax") 
    print("• Input: 3D CT volumes (1 channel)")
    print("• Output: Binary masks (1 channel)")
    print("• Challenge: Highly imbalanced (most voxels are background)")
    
    print(f"\nPotential issues causing loss → 0:")
    print("1. 🎯 Target format mismatch:")
    print("   - DiceCELoss expects specific target format")
    print("   - to_onehot_y=False means targets should be class indices")
    print("   - to_onehot_y=True means targets should be one-hot encoded")
    
    print("2. 🔢 Data type issues:")
    print("   - Targets converted to float32 but might need to be long/int")
    print("   - Model outputs might need different activation")
    
    print("3. ⚖️  Class imbalance:")
    print("   - Most voxels are background (class 0)")
    print("   - Very few pneumothorax voxels (class 1)")
    print("   - Model might predict all background → low loss")
    
    print("4. 🎛️  Loss function configuration:")
    print("   - sigmoid=True but model might already have sigmoid")
    print("   - CE part vs Dice part weighting")

def recommend_fixes():
    """Provide recommendations to fix the loss issue."""
    
    print(f"\n💡 Recommended Fixes:")
    print("=" * 30)
    
    print("1. 🔧 Check Model Output:")
    print("   Add debug prints to see model output range:")
    print("   print(f'Model output: min={outputs.min():.3f}, max={outputs.max():.3f}')")
    
    print("\n2. 🎯 Check Target Format:")
    print("   Add debug prints to see target properties:")
    print("   print(f'Targets: shape={targets.shape}, unique={torch.unique(targets)}')")
    
    print("\n3. 🔄 Try Different Loss Configurations:")
    print("   # Option A: Standard binary segmentation")
    print("   criterion = DiceCELoss(to_onehot_y=False, sigmoid=False)")
    print("   ")
    print("   # Option B: With logits")  
    print("   criterion = DiceLoss() + nn.BCEWithLogitsLoss()")
    print("   ")
    print("   # Option C: Weighted for class imbalance")
    print("   criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, include_background=False)")
    
    print("\n4. 🏥 Medical Imaging Specific:")
    print("   # Use Focal Loss for imbalanced data")
    print("   from monai.losses import FocalLoss")
    print("   criterion = FocalLoss()")
    print("   ")
    print("   # Or combine Dice + Focal")
    print("   dice_loss = DiceLoss(sigmoid=True)")
    print("   focal_loss = FocalLoss()")
    print("   # Then: loss = dice_loss(pred, target) + focal_loss(pred, target)")
    
    print("\n5. 🐛 Add Loss Debugging:")
    print("   In training loop, add these checks:")
    print("   ```python")
    print("   # Before loss calculation")
    print("   print(f'Outputs: {outputs.min():.3f} to {outputs.max():.3f}')")
    print("   print(f'Targets: {targets.unique()}')")
    print("   main_loss = criterion(outputs, targets)")
    print("   print(f'Loss: {main_loss.item():.6f}')")
    print("   if main_loss.item() < 0.001:")
    print("       print('WARNING: Loss too small!')")
    print("   ```")

def main():
    print("🚨 SIIM Loss Function Debugging Tool")
    print("=" * 50)
    
    test_dice_ce_loss()
    analyze_siim_data_properties()
    recommend_fixes()
    
    print(f"\n🎯 Immediate Action Items:")
    print("1. Add debug prints in your training loop")
    print("2. Check if model outputs and targets have expected ranges")
    print("3. Try DiceCELoss(to_onehot_y=False, sigmoid=False)")
    print("4. Consider using Focal Loss for class imbalance")
    print("5. Verify that labels are actually binary (0/1)")

if __name__ == "__main__":
    main()