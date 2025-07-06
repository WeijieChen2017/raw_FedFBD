#!/usr/bin/env python3
"""
Test SIIM data path resolution
"""

import os
import json
from argparse import Namespace

print("Testing SIIM data path resolution...")

# Create mock args similar to config
args = Namespace()
args.data_root = "./siim-101/SIIM_Fed_Learning_Phase1Data"
args.fold = 0
args.num_clients = 6

print(f"Original data_root: {args.data_root}")

# Test path resolution logic
data_root = getattr(args, 'data_root', './code_template/siim-101')

# If data_root ends with SIIM_Fed_Learning_Phase1Data, use its parent directory
if data_root.endswith('SIIM_Fed_Learning_Phase1Data'):
    data_root = os.path.dirname(data_root)

print(f"Resolved data_root: {data_root}")

# Load fold config
fold_config_path = f"config/siim/siim_fbd_fold_{args.fold}.json"
if os.path.exists(fold_config_path):
    with open(fold_config_path, 'r') as f:
        fold_config = json.load(f)
    
    print(f"Loaded fold config: {fold_config_path}")
    
    # Test path resolution for first client
    if 'train' in fold_config and 'client_0' in fold_config['train']:
        client_samples = fold_config['train']['client_0']
        first_sample = client_samples[0]
        
        print(f"\nOriginal sample paths:")
        print(f"  Image: {first_sample['image']}")
        print(f"  Label: {first_sample['label']}")
        
        # Resolve paths
        resolved_sample = {}
        for key, path in first_sample.items():
            if isinstance(path, str) and not os.path.isabs(path):
                resolved_path = os.path.join(data_root, path)
                resolved_sample[key] = resolved_path
            else:
                resolved_sample[key] = path
        
        print(f"\nResolved sample paths:")
        print(f"  Image: {resolved_sample['image']}")
        print(f"  Label: {resolved_sample['label']}")
        
        # Check if resolved paths exist
        print(f"\nPath existence check:")
        print(f"  Image exists: {os.path.exists(resolved_sample['image'])}")
        print(f"  Label exists: {os.path.exists(resolved_sample['label'])}")
        
        if not os.path.exists(resolved_sample['image']):
            print(f"\n❌ Image file not found!")
            print(f"Expected: {resolved_sample['image']}")
            
            # Check alternative paths
            alt_paths = [
                os.path.join('./code_template/siim-101', first_sample['image']),
                os.path.join('.', first_sample['image']),
                first_sample['image']
            ]
            
            print(f"\nChecking alternative paths:")
            for alt_path in alt_paths:
                exists = os.path.exists(alt_path)
                print(f"  {alt_path}: {'✓' if exists else '✗'}")
        else:
            print(f"\n✅ Path resolution working correctly!")
else:
    print(f"❌ Fold config not found: {fold_config_path}")

print("\nSIIM data path resolution test complete.")