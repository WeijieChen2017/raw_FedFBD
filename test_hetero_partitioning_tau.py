#!/usr/bin/env python3
"""
Test script for heterogeneous dataset partitioning
"""

import sys
sys.path.append('.')
import torch
import numpy as np
from fbd_main_tau import load_hetero_config, create_hetero_partitions
from fbd_dataset import load_data
from collections import Counter
import medmnist
from medmnist import INFO

def test_hetero_partitioning():
    """Test heterogeneous partitioning with OrganAMNIST"""
    
    # Mock args for testing
    class MockArgs:
        def __init__(self):
            self.experiment_name = 'organamnist'
            self.download = True
            self.as_rgb = False
            self.size = 28
            self.root = './cache'
            self.cache_dir = './cache'
            self.task = 'multi-class'
            self.num_classes = 11
    
    args = MockArgs()
    
    print("Testing Heterogeneous Dataset Partitioning")
    print("=" * 50)
    
    # Load hetero config
    hetero_config = load_hetero_config('organamnist_tau')
    if hetero_config is None:
        print("❌ Failed to load hetero config")
        return False
    
    print("✅ Loaded hetero config successfully")
    
    # Load dataset
    try:
        train_dataset, _ = load_data(args)
        print(f"✅ Loaded OrganAMNIST dataset: {len(train_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Create partitions
    try:
        partitions = create_hetero_partitions(train_dataset, hetero_config, seed=42)
        print(f"✅ Created {len(partitions)} client partitions")
    except Exception as e:
        print(f"❌ Failed to create partitions: {e}")
        return False
    
    # Validate partitions
    print("\nValidation Results:")
    print("-" * 30)
    
    total_samples = 0
    class_counts = {i: 0 for i in range(11)}
    
    for client_idx, partition in enumerate(partitions):
        # Count samples per client
        client_samples = len(partition)
        total_samples += client_samples
        
        # Count classes per client
        client_class_counts = Counter()
        for idx in range(client_samples):
            _, label = partition[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            client_class_counts[label] += 1
            class_counts[label] += 1
        
        # Display client distribution
        print(f"Client {client_idx}: {client_samples} samples")
        class_percentages = []
        for class_idx in range(11):
            class_count = client_class_counts[class_idx]
            percentage = (class_count / client_samples * 100) if client_samples > 0 else 0
            class_percentages.append(f"{percentage:.1f}%")
        print(f"  Class distribution: {class_percentages}")
    
    # Check total samples
    original_total = len(train_dataset)
    print(f"\nTotal samples check: {total_samples}/{original_total}")
    if total_samples == original_total:
        print("✅ All samples accounted for")
    else:
        print("❌ Sample count mismatch")
        return False
    
    # Check class distribution
    print(f"\nClass distribution across all clients:")
    for class_idx in range(11):
        count = class_counts[class_idx]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  Class {class_idx}: {count} samples ({percentage:.1f}%)")
    
    print("\n✅ Heterogeneous partitioning test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_hetero_partitioning()
    sys.exit(0 if success else 1)