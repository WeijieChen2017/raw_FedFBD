#!/usr/bin/env python3
"""
Generate SIIM FBD fold configurations for cross-validation.
Takes balanced_siim_4fold_128.json and creates 4 fold configurations.
"""

import json
import os
from collections import defaultdict

def load_balanced_folds(input_file):
    """Load the balanced 4-fold SIIM data."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def extract_site_from_path(image_path):
    """Extract site number from image path."""
    # Path format: "SIIM_Fed_Learning_Phase1Data/Site_X_Test/For_FedTraining/data/..."
    parts = image_path.split('/')
    site_part = parts[1]  # "Site_X_Test"
    site_num = int(site_part.split('_')[1])  # Extract X
    return site_num

def organize_by_client(fold_data):
    """Organize fold data by client (site)."""
    clients = defaultdict(list)
    
    for item in fold_data:
        site_num = extract_site_from_path(item['image'])
        client_id = f"client_{site_num - 1}"  # Site_1 -> client_0, Site_2 -> client_1, etc.
        clients[client_id].append(item)
    
    return dict(clients)

def generate_fold_config(balanced_data, test_fold_idx):
    """Generate a single fold configuration."""
    folds = [balanced_data[f'fold_{i}'] for i in range(4)]
    
    # Determine fold assignments
    val_fold_idx = (test_fold_idx + 1) % 4
    train_fold_indices = [i for i in range(4) if i not in [test_fold_idx, val_fold_idx]]
    
    # Combine training folds
    train_data = []
    for idx in train_fold_indices:
        train_data.extend(folds[idx])
    
    # Organize by client
    config = {
        'train': organize_by_client(train_data),
        'val': organize_by_client(folds[val_fold_idx]),
        'test': organize_by_client(folds[test_fold_idx])
    }
    
    return config

def main():
    # Input and output paths
    input_file = 'code_template/siim-101/balanced_siim_4fold_128.json'
    output_dir = 'config/siim'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load balanced fold data
    print(f"Loading balanced fold data from {input_file}")
    balanced_data = load_balanced_folds(input_file)
    
    # Generate all 4 fold configurations
    for fold_idx in range(4):
        print(f"Generating fold {fold_idx} configuration...")
        
        config = generate_fold_config(balanced_data, fold_idx)
        
        # Save configuration
        output_file = os.path.join(output_dir, f'siim_fbd_fold_{fold_idx}.json')
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved {output_file}")
        
        # Print summary
        train_total = sum(len(clients) for clients in config['train'].values())
        val_total = sum(len(clients) for clients in config['val'].values())
        test_total = sum(len(clients) for clients in config['test'].values())
        
        print(f"  Train: {train_total} samples across {len(config['train'])} clients")
        print(f"  Val: {val_total} samples across {len(config['val'])} clients")
        print(f"  Test: {test_total} samples across {len(config['test'])} clients")
        print()

if __name__ == '__main__':
    main() 