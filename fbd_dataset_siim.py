import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
    EnsureTyped,
)
from monai.data import PersistentDataset, CacheDataset
import nibabel as nib
import logging


class SIIMSegmentationDataset(Dataset):
    """Dataset class for SIIM segmentation data."""
    
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transforms:
            data = self.transforms(data)
        return data


def load_siim_fold_data(args):
    """Load SIIM fold data from JSON file."""
    fold_file_path = os.path.join(args.data_root, "..", args.fold_file)
    
    if not os.path.exists(fold_file_path):
        # Try looking in the code_template/siim-101 directory
        fold_file_path = os.path.join("code_template", "siim-101", args.fold_file)
    
    with open(fold_file_path, 'r') as f:
        fold_data = json.load(f)
    
    return fold_data


def filter_invalid_samples(data_list, min_z_dim=16, logger=None):
    """Filter out samples with z-dimension less than min_z_dim."""
    valid_data = []
    invalid_data = []
    
    for sample in data_list:
        # Check if the image file exists and has sufficient z-dimension
        image_path = sample.get("image", "")
        if os.path.exists(image_path):
            try:
                # Load header to check dimensions
                img = nib.load(image_path)
                z_dim = img.shape[2] if len(img.shape) >= 3 else 0
                
                if z_dim >= min_z_dim:
                    valid_data.append(sample)
                else:
                    invalid_data.append(sample)
                    if logger:
                        logger.debug(f"Filtered out {image_path}: z-dim {z_dim} < {min_z_dim}")
            except Exception as e:
                invalid_data.append(sample)
                if logger:
                    logger.warning(f"Error loading {image_path}: {e}")
        else:
            invalid_data.append(sample)
            if logger:
                logger.warning(f"Image not found: {image_path}")
    
    return valid_data, invalid_data


def load_siim_data(args):
    """
    Load SIIM dataset for training and testing.
    Returns train and test datasets.
    """
    # Load fold data
    fold_data = load_siim_fold_data(args)
    
    # Get training and validation data for the specified fold
    train_folds = [f"fold_{(args.fold+1) % 4}", f"fold_{(args.fold+2) % 4}"]
    val_fold = f"fold_{(args.fold+3) % 4}"
    test_fold = f"fold_{args.fold}"
    
    train_data = []
    for fold in train_folds:
        if fold in fold_data:
            train_data.extend(fold_data[fold])
    
    test_data = fold_data.get(test_fold, [])
    
    # Filter out invalid samples
    train_data, _ = filter_invalid_samples(train_data, min_z_dim=args.min_z_dim)
    test_data, _ = filter_invalid_samples(test_data, min_z_dim=args.min_z_dim)
    
    # Define transforms
    min_intensity = -1024
    max_intensity = 1976
    
    # Training transforms with augmentation
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.1),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Test transforms without augmentation
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], b_min=0.0, b_max=1.0, 
                           a_min=min_intensity, a_max=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create datasets with caching if specified
    if args.cache_type == "none":
        train_dataset = SIIMSegmentationDataset(train_data, transforms=train_transforms)
        test_dataset = SIIMSegmentationDataset(test_data, transforms=test_transforms)
    elif args.cache_type == "disk":
        os.makedirs(args.cache_dir, exist_ok=True)
        train_dataset = PersistentDataset(
            data=train_data,
            transform=train_transforms,
            cache_dir=os.path.join(args.cache_dir, "train")
        )
        test_dataset = PersistentDataset(
            data=test_data,
            transform=test_transforms,
            cache_dir=os.path.join(args.cache_dir, "test")
        )
    else:  # memory cache
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=args.num_workers
        )
        test_dataset = CacheDataset(
            data=test_data,
            transform=test_transforms,
            cache_rate=1.0,
            num_workers=args.num_workers
        )
    
    return train_dataset, test_dataset


def partition_siim_data(dataset, num_clients, iid=True):
    """
    Partition SIIM dataset among clients.
    For segmentation tasks, we partition by patient/study rather than by class.
    """
    total_samples = len(dataset)
    
    if iid:
        # Random partitioning
        indices = np.random.permutation(total_samples)
        partition_size = total_samples // num_clients
        
        partitions = []
        for i in range(num_clients):
            start_idx = i * partition_size
            if i == num_clients - 1:
                # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = start_idx + partition_size
            
            client_indices = indices[start_idx:end_idx].tolist()
            partitions.append(torch.utils.data.Subset(dataset, client_indices))
    else:
        # Non-IID partitioning - could be based on different criteria
        # For medical imaging, non-IID could mean different scanners, protocols, etc.
        # For now, we'll use a simple uneven split
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        # Create uneven partitions
        partition_sizes = []
        remaining = total_samples
        
        for i in range(num_clients - 1):
            # Random size between 10% and 30% of remaining
            min_size = max(1, int(0.1 * remaining))
            max_size = max(min_size, int(0.3 * remaining))
            size = np.random.randint(min_size, max_size + 1)
            partition_sizes.append(size)
            remaining -= size
        
        # Last client gets all remaining
        partition_sizes.append(remaining)
        
        partitions = []
        start_idx = 0
        for size in partition_sizes:
            end_idx = start_idx + size
            client_indices = indices[start_idx:end_idx]
            partitions.append(torch.utils.data.Subset(dataset, client_indices))
            start_idx = end_idx
    
    return partitions


def get_siim_data_loader(dataset_partition, batch_size, shuffle=True, num_workers=4):
    """Create a DataLoader for SIIM dataset partition."""
    return DataLoader(
        dataset_partition,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )