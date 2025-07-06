#!/usr/bin/env python3
"""
Analyze SIIM dataset sizes from JSON fold files
This script checks all images in the fold files and reports their dimensions
"""

import os
import json
import glob
import nibabel as nib
from collections import defaultdict, Counter
import argparse

def load_fold_config(fold_path):
    """Load fold configuration from JSON file."""
    try:
        with open(fold_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {fold_path}: {e}")
        return None

def get_image_size(image_path):
    """Get the size of a NIfTI image."""
    try:
        if os.path.exists(image_path):
            img = nib.load(image_path)
            return img.shape
        else:
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def analyze_dataset_sizes(data_root="./code_template/siim-101", config_dir="config/siim"):
    """Analyze sizes of all images in SIIM dataset."""
    
    print("=" * 80)
    print("SIIM Dataset Size Analysis")
    print("=" * 80)
    
    # Find all fold configuration files
    fold_files = glob.glob(os.path.join(config_dir, "siim_fbd_fold_*.json"))
    fold_files.sort()
    
    if not fold_files:
        print(f"No fold files found in {config_dir}")
        return
    
    print(f"Found {len(fold_files)} fold files:")
    for f in fold_files:
        print(f"  - {f}")
    
    # Adjust data_root if it ends with SIIM_Fed_Learning_Phase1Data
    if data_root.endswith('SIIM_Fed_Learning_Phase1Data'):
        data_root = os.path.dirname(data_root)
    
    print(f"\nUsing data root: {data_root}")
    
    all_image_sizes = []
    all_label_sizes = []
    size_distribution = Counter()
    failed_files = []
    total_samples = 0
    
    # Process each fold file
    for fold_file in fold_files:
        fold_name = os.path.basename(fold_file)
        print(f"\n--- Processing {fold_name} ---")
        
        fold_config = load_fold_config(fold_file)
        if not fold_config:
            continue
        
        # Process train and test data
        for split_name in ['train', 'test']:
            if split_name not in fold_config:
                continue
                
            print(f"\n{split_name.upper()} data:")
            split_data = fold_config[split_name]
            
            # Process each client
            for client_key, client_samples in split_data.items():
                print(f"  {client_key}: {len(client_samples)} samples")
                
                for sample_idx, sample in enumerate(client_samples):
                    total_samples += 1
                    
                    # Get image and label paths
                    image_rel_path = sample.get('image', '')
                    label_rel_path = sample.get('label', '')
                    
                    # Resolve full paths
                    image_full_path = os.path.join(data_root, image_rel_path)
                    label_full_path = os.path.join(data_root, label_rel_path)
                    
                    # Check image size
                    image_size = get_image_size(image_full_path)
                    if image_size:
                        all_image_sizes.append(image_size)
                        size_distribution[image_size] += 1
                        
                        # Quick check: report if not 128x128
                        if image_size[:2] != (128, 128):
                            print(f"    ⚠️  {client_key}[{sample_idx}]: Image size {image_size} (expected 128x128)")
                    else:
                        failed_files.append(('image', image_full_path))
                        print(f"    ❌ {client_key}[{sample_idx}]: Image not found - {image_full_path}")
                    
                    # Check label size
                    label_size = get_image_size(label_full_path)
                    if label_size:
                        all_label_sizes.append(label_size)
                        
                        # Check if label size matches image size
                        if image_size and label_size != image_size:
                            print(f"    ⚠️  {client_key}[{sample_idx}]: Label size {label_size} != Image size {image_size}")
                    else:
                        failed_files.append(('label', label_full_path))
                        print(f"    ❌ {client_key}[{sample_idx}]: Label not found - {label_full_path}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Successfully loaded images: {len(all_image_sizes)}")
    print(f"Successfully loaded labels: {len(all_label_sizes)}")
    print(f"Failed file loads: {len(failed_files)}")
    
    if all_image_sizes:
        print(f"\n--- Image Size Distribution ---")
        print(f"Unique image sizes found: {len(size_distribution)}")
        
        for size, count in size_distribution.most_common():
            percentage = (count / len(all_image_sizes)) * 100
            is_expected = "✓" if size[:2] == (128, 128) else "✗"
            print(f"  {is_expected} {size}: {count} images ({percentage:.1f}%)")
        
        # Check for 128x128 compliance
        expected_count = sum(count for size, count in size_distribution.items() if size[:2] == (128, 128))
        compliance_rate = (expected_count / len(all_image_sizes)) * 100
        
        print(f"\n--- 128x128 Compliance ---")
        print(f"Images with 128x128 dimensions: {expected_count}/{len(all_image_sizes)} ({compliance_rate:.1f}%)")
        
        if compliance_rate < 100:
            print(f"⚠️  WARNING: {100-compliance_rate:.1f}% of images are NOT 128x128!")
            print("This will cause the DataLoader batch stacking error.")
            
            print(f"\nNon-compliant sizes:")
            for size, count in size_distribution.items():
                if size[:2] != (128, 128):
                    print(f"  - {size}: {count} images")
        else:
            print("✅ All images are 128x128 compliant!")
    
    if failed_files:
        print(f"\n--- Failed File Loads ---")
        for file_type, file_path in failed_files[:10]:  # Show first 10
            print(f"  {file_type}: {file_path}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    return {
        'total_samples': total_samples,
        'successful_images': len(all_image_sizes),
        'successful_labels': len(all_label_sizes),
        'size_distribution': dict(size_distribution),
        'failed_files': failed_files,
        'compliance_rate': compliance_rate if all_image_sizes else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze SIIM dataset image sizes')
    parser.add_argument('--data_root', default='./code_template/siim-101',
                       help='Root directory containing SIIM data')
    parser.add_argument('--config_dir', default='config/siim',
                       help='Directory containing fold configuration files')
    parser.add_argument('--export_report', type=str, default=None,
                       help='Export detailed report to JSON file')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_dataset_sizes(args.data_root, args.config_dir)
    
    # Export report if requested
    if args.export_report:
        # Convert Counter to regular dict for JSON serialization
        export_data = {
            'analysis_results': results,
            'data_root': args.data_root,
            'config_dir': args.config_dir
        }
        
        with open(args.export_report, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\nDetailed report exported to: {args.export_report}")

if __name__ == "__main__":
    main()