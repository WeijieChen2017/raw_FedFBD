#!/usr/bin/env python3
"""
Balanced SIIM DataLoader that controls positive/negative sample ratios in each batch.
Based on the approach mentioned in code_template/siim.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
import random


class BalancedBatchSampler:
    """
    Fast batch sampler that balances positive/negative samples on-the-fly.
    Uses probabilistic sampling instead of pre-analyzing the entire dataset.
    """
    
    def __init__(self, dataset, batch_size, positive_ratio=0.4, max_positive_per_batch=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.max_positive_per_batch = max_positive_per_batch
        
        # Calculate target composition - handle batch_size=1 case
        if batch_size == 1:
            # For batch_size=1, use probabilistic sampling with 10% minimum positive ratio
            self.min_positive_ratio = max(0.1, positive_ratio)  # At least 10% positive
            self.positive_per_batch = 1  # Will be handled probabilistically
            self.negative_per_batch = 0
        else:
            self.positive_per_batch = min(max_positive_per_batch, max(1, int(batch_size * positive_ratio)))
            self.negative_per_batch = batch_size - self.positive_per_batch
        
        # Estimate number of batches (will be approximate)
        self.num_batches = len(dataset) // batch_size
        
        print(f"📊 Balanced Sampler configured:")
        if batch_size == 1:
            print(f"   Single sample batches with minimum {self.min_positive_ratio*100:.1f}% positive probability")
        else:
            print(f"   Target batch composition: {self.positive_per_batch} positive + {self.negative_per_batch} negative")
        print(f"   Estimated batches per epoch: {self.num_batches}")
    
    def __iter__(self):
        """Generate balanced batches on-the-fly."""
        dataset_indices = list(range(len(self.dataset)))
        random.shuffle(dataset_indices)
        
        batch_count = 0
        idx = 0
        
        if self.batch_size == 1:
            # Special handling for batch_size=1: probabilistic positive sampling
            while batch_count < self.num_batches and idx < len(dataset_indices):
                # Scan for positive and negative samples
                positive_indices = []
                negative_indices = []
                scan_start = idx
                scan_limit = min(len(dataset_indices), idx + 50)  # Scan next 50 samples
                
                for scan_idx in range(scan_start, scan_limit):
                    current_idx = dataset_indices[scan_idx % len(dataset_indices)]
                    try:
                        sample = self.dataset[current_idx]
                        if isinstance(sample, dict):
                            label = sample["label"]
                        else:
                            _, label = sample
                        
                        if torch.is_tensor(label):
                            has_positive = (label > 0.5).any().item()
                        else:
                            has_positive = (label > 0.5).any()
                        
                        if has_positive:
                            positive_indices.append(current_idx)
                        else:
                            negative_indices.append(current_idx)
                    except Exception:
                        negative_indices.append(current_idx)
                
                # Probabilistic selection: prefer positive samples with min_positive_ratio probability
                if positive_indices and (random.random() < self.min_positive_ratio or len(negative_indices) == 0):
                    yield [random.choice(positive_indices)]
                elif negative_indices:
                    yield [random.choice(negative_indices)]
                else:
                    # Fallback to next available sample
                    yield [dataset_indices[idx % len(dataset_indices)]]
                
                idx += 1
                batch_count += 1
        else:
            # Original multi-sample batch logic
            while batch_count < self.num_batches and idx < len(dataset_indices):
                batch_indices = []
                positive_count = 0
                negative_count = 0
                
                # Try to fill batch with balanced samples
                attempts = 0
                while len(batch_indices) < self.batch_size and idx < len(dataset_indices) and attempts < self.batch_size * 3:
                    current_idx = dataset_indices[idx % len(dataset_indices)]
                    
                    try:
                        # Quick check if sample is positive or negative
                        sample = self.dataset[current_idx]
                        if isinstance(sample, dict):
                            label = sample["label"]
                        else:
                            _, label = sample
                        
                        # Fast check for positive voxels
                        if torch.is_tensor(label):
                            has_positive = (label > 0.5).any().item()
                        else:
                            has_positive = (label > 0.5).any()
                        
                        # Add sample if it fits our balance requirements
                        if has_positive and positive_count < self.positive_per_batch:
                            batch_indices.append(current_idx)
                            positive_count += 1
                        elif not has_positive and negative_count < self.negative_per_batch:
                            batch_indices.append(current_idx)
                            negative_count += 1
                        
                    except Exception:
                        # If we can't process, treat as negative
                        if negative_count < self.negative_per_batch:
                            batch_indices.append(current_idx)
                            negative_count += 1
                    
                    idx += 1
                    attempts += 1
                
                # If we couldn't fill the batch perfectly, add remaining samples
                while len(batch_indices) < self.batch_size and idx < len(dataset_indices):
                    batch_indices.append(dataset_indices[idx])
                    idx += 1
                
                if len(batch_indices) == self.batch_size:
                    random.shuffle(batch_indices)  # Shuffle within batch
                    yield batch_indices
                    batch_count += 1
    
    def __len__(self):
        return self.num_batches
    
def get_balanced_siim_data_loader(dataset, batch_size, positive_ratio=0.4, max_positive_per_batch=2, shuffle=True):
    """
    Create a balanced DataLoader for SIIM dataset.
    
    Args:
        dataset: SIIM dataset
        batch_size: Batch size
        positive_ratio: Target ratio of positive samples (0.0 to 1.0)
        max_positive_per_batch: Maximum positive samples per batch
        shuffle: Whether to shuffle (uses balanced sampling)
    
    Returns:
        DataLoader with balanced sampling
    """
    if shuffle:
        # Use balanced batch sampler
        batch_sampler = BalancedBatchSampler(
            dataset, 
            batch_size, 
            positive_ratio=positive_ratio,
            max_positive_per_batch=max_positive_per_batch
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,  # Disable multiprocessing for custom sampler
            pin_memory=True
        )
    else:
        # Standard DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )


# Test function
def test_balanced_loader():
    """Test the balanced data loader with synthetic data."""
    import torch
    from torch.utils.data import TensorDataset
    
    # Create synthetic dataset with 10% positive samples
    num_samples = 100
    images = torch.randn(num_samples, 1, 32, 32, 16)
    
    # Create labels: 10% positive, 90% negative
    labels = torch.zeros(num_samples, 1, 32, 32, 16)
    positive_indices = np.random.choice(num_samples, size=10, replace=False)
    for idx in positive_indices:
        # Add some positive voxels
        labels[idx, 0, 10:20, 10:20, 5:10] = 1.0
    
    dataset = TensorDataset(images, labels)
    
    # Test balanced loader
    balanced_loader = get_balanced_siim_data_loader(
        dataset, 
        batch_size=4, 
        positive_ratio=0.5,
        max_positive_per_batch=2
    )
    
    print("Testing balanced loader:")
    for i, (batch_images, batch_labels) in enumerate(balanced_loader):
        positive_samples = sum((label > 0.5).any() for label in batch_labels)
        print(f"Batch {i}: {positive_samples}/{len(batch_labels)} positive samples")
        if i >= 3:  # Test first few batches
            break


if __name__ == "__main__":
    test_balanced_loader()