import os
import shutil
import hashlib
import medmnist
import torch
import torchvision.transforms as transforms
import PIL
from torch.utils.data import DataLoader, Subset, random_split
from medmnist import INFO
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from medmnist import INFO, Evaluator
from collections import defaultdict

CACHE_DIR = "../medmnist-101/data_storage"
MEDMNIST_DIR = os.path.expanduser("/root/.medmnist")

# From fbd_server.py
DATASET_SPECIFIC_RULES = {
    "breastmnist": {"as_rgb": True},
    "octmnist": {"as_rgb": True},
    "organcmnist": {"as_rgb": True},
    "tissuemnist": {"as_rgb": True},
    "pneumoniamnist": {"as_rgb": True},
    "chestmnist": {"as_rgb": True},
    "organamnist": {"as_rgb": True},
    "organsmnist": {"as_rgb": True},
}

def handle_dataset_cache(dataset, post_execution=False):
    """Manages the dataset cache by copying files when needed."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(MEDMNIST_DIR):
        os.makedirs(MEDMNIST_DIR)
        
    source_npz_path = os.path.join(CACHE_DIR, f"{dataset}.npz")
    dest_npz_path = os.path.join(MEDMNIST_DIR, f"{dataset}.npz")

    if not post_execution:
        print(f"Looking for {dataset} in {CACHE_DIR}")
        # Before execution: if file is in cache but not in destination, copy it.
        if os.path.exists(source_npz_path):
            if not os.path.exists(dest_npz_path):
                print(f"Found {dataset} in cache. Copying to {MEDMNIST_DIR}")
                shutil.copy(source_npz_path, dest_npz_path)
            else:
                # please have the md5 check here
                source_md5 = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                dest_md5 = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if source_md5 == dest_md5:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} and is the same")
                else:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} but is different, overwriting.")
                    shutil.copy(source_npz_path, dest_npz_path)
        else:
            print(f"Dataset {dataset} not found in cache {CACHE_DIR}")
    else:
        # Copy downloaded dataset to cache if not already there
        if os.path.exists(dest_npz_path):
            if not os.path.exists(source_npz_path):
                print(f"Copying {dataset} from {MEDMNIST_DIR} to cache")
                shutil.copy(dest_npz_path, source_npz_path)
            else:
                source_md5 = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                dest_md5 = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if source_md5 == dest_md5:
                    print(f"Dataset {dataset} already exists in cache and is the same")
                else:
                    print(f"Dataset {dataset} already exists in cache but is different, overwriting.")
                    shutil.copy(dest_npz_path, source_npz_path)

def load_data(args):
    """
    Loads the centralized training and test datasets from MedMNIST.

    Args:
        args: Command-line arguments containing dataset and model info.

    Returns:
        A tuple containing the training dataset and the test dataset.
    """
    data_flag = args.experiment_name
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Use as_rgb setting from config instead of hardcoded rules
    as_rgb = getattr(args, 'as_rgb', False)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=as_rgb, root=args.cache_dir)
    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=as_rgb, root=args.cache_dir)

    print(f"Loaded {data_flag}: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")
    return train_dataset, test_dataset

def partition_data(dataset, num_clients, iid=False):
    """
    Partitions the dataset for a number of clients.

    Args:
        dataset: The full dataset to partition.
        num_clients (int): The number of clients to partition for.
        iid (bool): Whether to perform an IID (stratified) or non-IID split.

    Returns:
        A list of Subset objects, one for each client.
    """
    num_samples = len(dataset)
    partitions = []
    
    if iid:
        print("Performing IID (stratified) data partition.")
        labels = dataset.labels if hasattr(dataset, 'labels') else [label for _, label in dataset]
        labels = np.array(labels).flatten()
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Distribute indices for each class among clients
        client_indices = [[] for _ in range(num_clients)]
        for cls_indices in class_indices.values():
            # Shuffle indices to ensure randomness
            np.random.shuffle(cls_indices)
            # Split shuffled indices among clients
            for i, idx in enumerate(cls_indices):
                client_indices[i % num_clients].append(idx)
        
        for i in range(num_clients):
            partitions.append(Subset(dataset, client_indices[i]))

    else:
        print("Performing non-IID data partition.")
        samples_per_client = num_samples // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            # For the last client, give it the remainder of the dataset
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
            indices = list(range(start_idx, end_idx))
            partitions.append(Subset(dataset, indices))

    print(f"Partitioned data into {len(partitions)} sets.")
    return partitions


def get_data_loader(dataset_partition, batch_size, num_workers=0):
    """
    Creates a DataLoader for a specific dataset partition.

    Args:
        dataset_partition: A Subset object representing the client's data.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of worker processes for the DataLoader.

    Returns:
        A PyTorch DataLoader.
    """
    return DataLoader(dataset_partition, batch_size=batch_size, shuffle=True, num_workers=num_workers) 