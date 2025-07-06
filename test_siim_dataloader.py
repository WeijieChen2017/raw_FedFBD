#!/usr/bin/env python3
"""
Test SIIM data loader format
"""

import torch
import os
import json
from argparse import Namespace
from fbd_dataset_siim import SIIMSegmentationDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CenterSpatialCropd, ToTensord, EnsureTyped
)

print("Testing SIIM data loader format...")

# Create mock sample data
mock_sample = {
    "image": "test_image_path.nii.gz",
    "label": "test_label_path.nii.gz"
}

# Create simple transforms that don't require actual files
# We'll mock the LoadImaged transform to avoid file loading
class MockLoadImaged:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        # Return mock tensors instead of loading files
        result = data.copy()
        for key in self.keys:
            if key == "image":
                # Mock 3D image tensor (C, H, W, D)
                result[key] = torch.randn(1, 64, 64, 32)
            elif key == "label":
                # Mock 3D label tensor (C, H, W, D) with binary values
                result[key] = torch.randint(0, 2, (1, 64, 64, 32)).float()
        return result

# Define simple test transforms
test_transforms = Compose([
    MockLoadImaged(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

print("Creating test dataset...")
test_data = [mock_sample]
dataset = SIIMSegmentationDataset(test_data, transforms=test_transforms)

print(f"Dataset length: {len(dataset)}")

print("\nTesting dataset __getitem__...")
try:
    sample_output = dataset[0]
    print(f"Output type: {type(sample_output)}")
    
    if isinstance(sample_output, tuple):
        image, label = sample_output
        print(f"✅ Dataset returns tuple format")
        print(f"  Image type: {type(image)}")
        print(f"  Image shape: {image.shape}")
        print(f"  Label type: {type(label)}")
        print(f"  Label shape: {label.shape}")
        
        # Test that tensors have the right methods
        try:
            image.to('cpu')
            label.to('cpu')
            print("✅ Tensors have .to() method")
        except Exception as e:
            print(f"❌ Tensor .to() method error: {e}")
            
    else:
        print(f"❌ Dataset returns {type(sample_output)}, expected tuple")
        
except Exception as e:
    print(f"❌ Dataset __getitem__ error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting DataLoader...")
try:
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"✅ DataLoader iteration {batch_idx}")
        print(f"  Inputs type: {type(inputs)}")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets type: {type(targets)}")
        print(f"  Targets shape: {targets.shape}")
        
        # Test the exact operation that was failing
        try:
            device = torch.device('cpu')
            inputs_on_device = inputs.to(device)
            print("✅ inputs.to(device) works correctly")
        except Exception as e:
            print(f"❌ inputs.to(device) error: {e}")
        
        break  # Only test first batch
        
except Exception as e:
    print(f"❌ DataLoader error: {e}")
    import traceback
    traceback.print_exc()

print("\nSIIM data loader format test complete.")