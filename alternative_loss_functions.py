#!/usr/bin/env python3
"""
Alternative loss functions for SIIM pneumothorax segmentation.
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss

def get_siim_loss_function(loss_type="dice_ce", device=None):
    """
    Get appropriate loss function for SIIM segmentation.
    
    Args:
        loss_type (str): Type of loss function to use
        device: Device to place tensors on (for weighted losses)
    """
    
    if loss_type == "dice_ce":
        from monai.losses import DiceCELoss
        return DiceCELoss(to_onehot_y=False, sigmoid=True)
    
    elif loss_type == "dice_only":
        return DiceLoss(sigmoid=True)
    
    elif loss_type == "bce_only":
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    
    elif loss_type == "dice_focal":
        return DiceFocalLoss(sigmoid=True, focal_weight=1.0, dice_weight=1.0)
    
    elif loss_type == "weighted_bce":
        # Weight for extreme class imbalance in SIIM (99% background, 1% foreground)
        pos_weight = torch.tensor([99.0])
        if device is not None:
            pos_weight = pos_weight.to(device)
        print(f"🎯 Using weighted BCE with pos_weight={pos_weight.item():.1f} on device {device}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_type == "extreme_weighted_bce":
        # Extremely aggressive weights for SIIM's extreme imbalance
        pos_weight = torch.tensor([500.0])
        if device is not None:
            pos_weight = pos_weight.to(device)
        print(f"🎯 Using EXTREME weighted BCE with pos_weight={pos_weight.item():.1f} on device {device}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_type == "dice_ce_no_sigmoid":
        # Try DiceCE without internal sigmoid
        from monai.losses import DiceCELoss
        return DiceCELoss(to_onehot_y=False, sigmoid=False)
    
    elif loss_type == "tversky":
        # Tversky loss (good for imbalanced data)
        from monai.losses import TverskyLoss
        return TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    
    elif loss_type == "combined":
        # Custom combination of losses
        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(sigmoid=True)
                self.bce = nn.BCEWithLogitsLoss()
                
            def forward(self, pred, target):
                dice_loss = self.dice(pred, target)
                bce_loss = self.bce(pred, target)
                return dice_loss + 0.5 * bce_loss
                
        return CombinedLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def test_loss_functions():
    """Test different loss functions with sample data."""
    
    print("🧪 Testing Alternative Loss Functions")
    print("=" * 50)
    
    # Create sample data
    batch_size, channels, h, w, d = 2, 1, 64, 64, 16
    predictions = torch.randn(batch_size, channels, h, w, d)
    targets = torch.zeros(batch_size, channels, h, w, d)
    targets[0, 0, 20:40, 20:40, 5:10] = 1.0  # Add some positive regions
    
    loss_types = ["dice_ce", "dice_only", "bce_only", "focal", "dice_focal", "weighted_bce", "combined"]
    
    for loss_type in loss_types:
        try:
            criterion = get_siim_loss_function(loss_type)
            loss = criterion(predictions, targets)
            print(f"{loss_type:15s}: {loss.item():.6f}")
        except Exception as e:
            print(f"{loss_type:15s}: ERROR - {e}")

if __name__ == "__main__":
    test_loss_functions()