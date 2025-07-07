#!/usr/bin/env python3
"""
Test script to verify the fold configuration loading works correctly.
"""
import json
import os

def load_fold_config(fold_idx):
    """Load fold configuration from saved JSON files."""
    fold_config_path = f"config/siim/siim_fbd_fold_{fold_idx}.json"
    
    if not os.path.exists(fold_config_path):
        raise FileNotFoundError(f"Fold configuration file not found: {fold_config_path}")
    
    with open(fold_config_path, 'r') as f:
        fold_config = json.load(f)
    
    return fold_config

def test_fold_config_loading():
    """Test loading all fold configurations"""
    print("Testing fold configuration loading...")
    
    for fold_idx in range(4):
        print(f"\nTesting fold {fold_idx}:")
        try:
            fold_config = load_fold_config(fold_idx)
            
            # Check structure
            required_keys = ['train', 'val', 'test']
            for key in required_keys:
                if key not in fold_config:
                    print(f"  ERROR: Missing key '{key}' in fold configuration")
                    continue
                    
                # Count clients and samples
                clients = fold_config[key]
                num_clients = len(clients)
                total_samples = sum(len(client_data) for client_data in clients.values())
                
                print(f"  {key}: {num_clients} clients, {total_samples} samples")
                
                # Show distribution per client
                for client_id, client_data in clients.items():
                    print(f"    {client_id}: {len(client_data)} samples")
                    
            print(f"  SUCCESS: Fold {fold_idx} loaded successfully")
            
        except Exception as e:
            print(f"  ERROR: Failed to load fold {fold_idx}: {e}")
    
    # Test invalid fold
    print(f"\nTesting invalid fold:")
    try:
        fold_config = load_fold_config(5)
        print(f"  ERROR: Should have failed to load invalid fold")
    except FileNotFoundError as e:
        print(f"  SUCCESS: Correctly rejected invalid fold: {e}")

if __name__ == "__main__":
    test_fold_config_loading() 