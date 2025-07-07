#!/usr/bin/env python3
"""
Script to analyze and fix SIIM data normalization issues.
"""

import torch
import numpy as np
import nibabel as nib
import os
import sys
from pathlib import Path

def analyze_siim_intensity_range(data_root="./siim-101/SIIM_Fed_Learning_Phase1Data"):
    """Analyze actual intensity ranges in SIIM data to determine optimal normalization."""
    
    print("🔍 Analyzing SIIM intensity ranges...")
    
    if not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        return None
    
    # Find sample images
    image_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.nii.gz') and 'image' in file.lower():
                image_files.append(os.path.join(root, file))
                if len(image_files) >= 10:  # Sample first 10 images
                    break
        if len(image_files) >= 10:
            break
    
    if not image_files:
        print(f"❌ No image files found in {data_root}")
        return None
    
    print(f"📊 Analyzing {len(image_files)} sample images...")
    
    all_intensities = []
    for img_path in image_files:
        try:
            img = nib.load(img_path)
            data = img.get_fdata()
            all_intensities.extend(data.flatten())
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
    
    if not all_intensities:
        print("❌ No valid intensity data found")
        return None
    
    all_intensities = np.array(all_intensities)
    
    # Calculate statistics
    stats = {
        'min': np.min(all_intensities),
        'max': np.max(all_intensities),
        'mean': np.mean(all_intensities),
        'std': np.std(all_intensities),
        'q1': np.percentile(all_intensities, 25),
        'median': np.percentile(all_intensities, 50),
        'q3': np.percentile(all_intensities, 75),
        'p5': np.percentile(all_intensities, 5),
        'p95': np.percentile(all_intensities, 95),
        'p1': np.percentile(all_intensities, 1),
        'p99': np.percentile(all_intensities, 99)
    }
    
    print(f"\n📈 Intensity Statistics:")
    print(f"   Min: {stats['min']:.1f}")
    print(f"   Max: {stats['max']:.1f}")
    print(f"   Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
    print(f"   Median: {stats['median']:.1f}")
    print(f"   Q1-Q3: {stats['q1']:.1f} - {stats['q3']:.1f}")
    print(f"   P1-P99: {stats['p1']:.1f} - {stats['p99']:.1f}")
    print(f"   P5-P95: {stats['p5']:.1f} - {stats['p95']:.1f}")
    
    return stats

def get_optimal_normalization_params(stats):
    """Determine optimal normalization parameters based on data statistics."""
    
    if stats is None:
        print("📌 Using standard CT lung window defaults")
        return {
            'method': 'lung_window',
            'min_intensity': -1000,  # Standard lung window
            'max_intensity': 400,
            'reason': 'Standard lung window for pneumothorax detection'
        }
    
    # Lung tissue typically ranges from -1000 (air) to +50 (soft tissue)
    # Pneumothorax appears as air (-1000 HU) in abnormal locations
    
    # Option 1: Conservative lung window
    conservative = {
        'method': 'conservative_lung',
        'min_intensity': -1000,
        'max_intensity': 200,
        'reason': 'Conservative lung window focusing on air and soft tissue'
    }
    
    # Option 2: Data-driven approach using percentiles
    data_driven = {
        'method': 'data_driven',
        'min_intensity': stats['p1'],
        'max_intensity': stats['p99'],
        'reason': 'Based on 1st-99th percentile of actual data'
    }
    
    # Option 3: Robust approach using IQR
    iqr_factor = 1.5
    iqr = stats['q3'] - stats['q1']
    robust = {
        'method': 'robust_iqr',
        'min_intensity': max(-1024, stats['q1'] - iqr_factor * iqr),
        'max_intensity': min(3000, stats['q3'] + iqr_factor * iqr),
        'reason': 'Robust approach using IQR with outlier protection'
    }
    
    print(f"\n🎯 Normalization Recommendations:")
    print(f"1. Conservative Lung Window: [{conservative['min_intensity']:.0f}, {conservative['max_intensity']:.0f}]")
    print(f"2. Data-Driven (P1-P99): [{data_driven['min_intensity']:.0f}, {data_driven['max_intensity']:.0f}]")
    print(f"3. Robust IQR: [{robust['min_intensity']:.0f}, {robust['max_intensity']:.0f}]")
    
    # Choose the most appropriate
    if stats['max'] > 2000:  # High contrast data
        recommended = conservative
    elif abs(data_driven['max_intensity'] - data_driven['min_intensity']) < 500:
        recommended = robust
    else:
        recommended = data_driven
    
    print(f"\n✅ Recommended: {recommended['method']} - {recommended['reason']}")
    return recommended

def create_improved_transforms():
    """Create improved MONAI transforms with better normalization."""
    
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
        CenterSpatialCropd, RandFlipd, RandRotate90d, RandShiftIntensityd,
        RandScaleIntensityd, ToTensord, EnsureTyped, DivisiblePadd,
        NormalizeIntensityd, ThresholdIntensityd
    )
    
    # Recommended normalization parameters
    min_intensity = -1000  # Lung window
    max_intensity = 400
    
    print(f"🔧 Creating improved transforms with lung window: [{min_intensity}, {max_intensity}]")
    
    # Training transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Step 1: Clip extreme outliers first
        ThresholdIntensityd(keys=["image"], threshold=min_intensity, above=True, cval=min_intensity),
        ThresholdIntensityd(keys=["image"], threshold=max_intensity, above=False, cval=max_intensity),
        
        # Step 2: Normalize to [0, 1] range
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        
        # Step 3: Ensure label is binary
        ThresholdIntensityd(keys=["label"], threshold=0.5, above=True, cval=1.0),
        ThresholdIntensityd(keys=["label"], threshold=0.5, above=False, cval=0.0),
        
        # Spatial transforms
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 32]),
        DivisiblePadd(keys=["image", "label"], k=16),
        
        # Data augmentation (reduced intensity for stability)
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, spatial_axes=(0, 1)),
        
        # Reduced intensity augmentation for medical images
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.05),
        RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.05),
        
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Test transforms (no augmentation)
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Same normalization as training
        ThresholdIntensityd(keys=["image"], threshold=min_intensity, above=True, cval=min_intensity),
        ThresholdIntensityd(keys=["image"], threshold=max_intensity, above=False, cval=max_intensity),
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        ThresholdIntensityd(keys=["label"], threshold=0.5, above=True, cval=1.0),
        ThresholdIntensityd(keys=["label"], threshold=0.5, above=False, cval=0.0),
        
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 32]),
        DivisiblePadd(keys=["image", "label"], k=16),
        
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    return train_transforms, test_transforms

def save_improved_normalization_config():
    """Save improved normalization configuration."""
    
    config = {
        "normalization": {
            "method": "lung_window",
            "min_intensity": -1000,
            "max_intensity": 400,
            "description": "Optimized for pneumothorax detection in lung CT"
        },
        "augmentation": {
            "flip_prob": 0.3,
            "rotation_prob": 0.1,
            "intensity_shift_prob": 0.3,
            "intensity_shift_offsets": 0.05,
            "intensity_scale_prob": 0.3,
            "intensity_scale_factors": 0.05
        },
        "spatial": {
            "roi_size": [128, 128, 32],
            "padding_divisible": 16
        }
    }
    
    import json
    with open("siim_improved_normalization.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("💾 Saved improved normalization config to siim_improved_normalization.json")
    return config

def main():
    print("🔧 SIIM Data Normalization Analysis and Fix")
    print("=" * 60)
    
    # Try to analyze actual data
    stats = analyze_siim_intensity_range()
    
    # Get recommendations
    params = get_optimal_normalization_params(stats)
    
    # Create improved transforms
    train_transforms, test_transforms = create_improved_transforms()
    
    # Save configuration
    config = save_improved_normalization_config()
    
    print(f"\n🎯 Recommendations:")
    print(f"1. Use lung window normalization: [-1000, 400] HU")
    print(f"2. Ensure binary labels with threshold at 0.5")
    print(f"3. Add DivisiblePadd for consistent dimensions")
    print(f"4. Reduce augmentation intensity for medical images")
    print(f"5. Update batch size to 4 for better gradient estimates")
    
    print(f"\n📝 Next Steps:")
    print(f"1. Update fbd_dataset_siim.py with improved transforms")
    print(f"2. Test with small model first: --model_size small")
    print(f"3. Monitor input/output ranges during training")
    print(f"4. Check for NaN losses (current debug code will catch this)")

if __name__ == "__main__":
    main()