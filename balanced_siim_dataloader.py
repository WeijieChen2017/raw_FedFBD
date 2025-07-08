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


class BalancedSampler(Sampler):
    """
    Sampler that ensures each batch has a controlled ratio of positive/negative samples.
    
    Args:
        dataset: SIIM dataset with binary labels
        batch_size: Size of each batch
        positive_ratio: Ratio of positive samples in each batch (0.0 to 1.0)
        max_positive_per_batch: Maximum number of positive samples per batch
    """
    
    def __init__(self, dataset, batch_size, positive_ratio=0.5, max_positive_per_batch=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        
        # Calculate positive samples per batch
        if max_positive_per_batch is None:
            self.positive_per_batch = max(1, int(batch_size * positive_ratio))
        else:
            self.positive_per_batch = min(max_positive_per_batch, int(batch_size * positive_ratio))
        
        self.negative_per_batch = batch_size - self.positive_per_batch
        
        # Categorize samples by positive/negative labels
        self.positive_indices = []
        self.negative_indices = []
        
        print(f"🔍 Analyzing dataset to categorize positive/negative samples...")
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                if isinstance(sample, dict):
                    label = sample["label"]
                else:
                    _, label = sample
                
                # Check if sample has pneumothorax (positive voxels)
                if torch.is_tensor(label):
                    has_positive = (label > 0.5).any().item()
                else:
                    has_positive = (label > 0.5).any()
                
                if has_positive:
                    self.positive_indices.append(idx)
                else:
                    self.negative_indices.append(idx)
                    
            except Exception as e:
                print(f"Warning: Could not process sample {idx}: {e}")
                # Default to negative if we can't determine
                self.negative_indices.append(idx)
        
        print(f"✅ Dataset analysis complete:")
        print(f"   Positive samples: {len(self.positive_indices)}")
        print(f"   Negative samples: {len(self.negative_indices)}")
        print(f"   Batch composition: {self.positive_per_batch} positive + {self.negative_per_batch} negative")
        
        # Ensure we have enough samples
        if len(self.positive_indices) == 0:
            print("⚠️  WARNING: No positive samples found! Using random sampling.")
            self.use_balanced = False
        elif len(self.negative_indices) == 0:
            print("⚠️  WARNING: No negative samples found! Using random sampling.")
            self.use_balanced = False
        else:
            self.use_balanced = True
        
        # Calculate number of batches
        if self.use_balanced:
            # Limited by the smaller class (with repetition)
            max_batches_pos = len(self.positive_indices) // max(1, self.positive_per_batch)
            max_batches_neg = len(self.negative_indices) // max(1, self.negative_per_batch)
            self.num_batches = min(max_batches_pos * 3, max_batches_neg)  # Allow 3x repetition of positive
        else:
            self.num_batches = len(self.dataset) // self.batch_size
    
    def __iter__(self):
        if not self.use_balanced:
            # Fallback to random sampling
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                yield indices[i:i + self.batch_size]
            return
        
        # Shuffle indices for each epoch
        positive_shuffled = self.positive_indices.copy()
        negative_shuffled = self.negative_indices.copy()
        random.shuffle(positive_shuffled)
        random.shuffle(negative_shuffled)
        
        # Create balanced batches
        pos_idx = 0
        neg_idx = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add positive samples
            for _ in range(self.positive_per_batch):
                batch_indices.append(positive_shuffled[pos_idx % len(positive_shuffled)])
                pos_idx += 1
            
            # Add negative samples
            for _ in range(self.negative_per_batch):
                batch_indices.append(negative_shuffled[neg_idx % len(negative_shuffled)])
                neg_idx += 1
            
            # Shuffle batch to avoid position bias
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


def get_balanced_siim_data_loader(dataset, batch_size, positive_ratio=0.3, max_positive_per_batch=1, shuffle=True):
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
    if shuffle and batch_size > 1:
        # Use balanced sampler
        sampler = BalancedSampler(
            dataset, 
            batch_size, 
            positive_ratio=positive_ratio,
            max_positive_per_batch=max_positive_per_batch
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
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