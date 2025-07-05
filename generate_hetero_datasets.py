#!/usr/bin/env python3
"""
Heterogeneous Dataset Distribution Generator for Multiple MedMNIST Datasets
Generates client-specific class distributions for federated learning with data heterogeneity.
"""

import json
import numpy as np
import random
import os
from typing import Dict, List

# Dataset configurations for MedMNIST datasets
DATASET_CONFIGS = {
    "bloodmnist": {
        "num_classes": 8,
        "class_names": ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]
    },
    "breastmnist": {
        "num_classes": 2,
        "class_names": ["malignant", "benign"]
    },
    "chestmnist": {
        "num_classes": 14,
        "class_names": ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia",
                       "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis", "pleural_thickening", "hernia"]
    },
    "dermamnist": {
        "num_classes": 7,
        "class_names": ["actinic keratoses", "basal cell carcinoma", "benign keratosis-like lesions", 
                       "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"]
    },
    "octmnist": {
        "num_classes": 4,
        "class_names": ["choroidal neovascularization", "diabetic macular edema", "drusen", "normal"]
    },
    "organamnist": {
        "num_classes": 11,
        "class_names": ["bladder", "femur-left", "femur-right", "heart", "kidney-left", 
                       "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
    },
    "organcmnist": {
        "num_classes": 11,
        "class_names": ["bladder", "femur-left", "femur-right", "heart", "kidney-left", 
                       "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
    },
    "organsmnist": {
        "num_classes": 11,
        "class_names": ["bladder", "femur-left", "femur-right", "heart", "kidney-left", 
                       "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"]
    },
    "pathmnist": {
        "num_classes": 9,
        "class_names": ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle", 
                       "normal_colon_mucosa", "cancer-associated_stroma", "colorectal_adenocarcinoma_epithelium"]
    },
    "pneumoniamnist": {
        "num_classes": 2,
        "class_names": ["normal", "pneumonia"]
    },
    "retinamnist": {
        "num_classes": 5,
        "class_names": ["0", "1", "2", "3", "4"]  # Disease severity grades
    },
    "tissuemnist": {
        "num_classes": 8,
        "class_names": ["collecting_duct_connecting_tubule", "distal_convoluted_tubule", "glomerular_tuft", 
                       "interstitial_endothelium", "interstitium", "proximal_tubule", "thick_ascending_limb", "vascular_pole"]
    }
}

def generate_dirichlet_distribution(num_clients: int, num_classes: int, alpha: float = 1.0) -> List[List[float]]:
    """
    Generate heterogeneous data distribution using Dirichlet distribution.
    
    Args:
        num_clients: Number of clients (6)
        num_classes: Number of classes (varies by dataset)
        alpha: Dirichlet concentration parameter (lower = more heterogeneous, default: 1.0)
    
    Returns:
        List of distributions for each client, where each distribution is a list of percentages
    """
    # Generate Dirichlet distribution for each class across clients
    class_distributions = []
    
    for class_idx in range(num_classes):
        # Generate client proportions for this class using Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Convert to percentages and ensure they sum to 100%
        percentages = (proportions * 100).tolist()
        class_distributions.append(percentages)
    
    # Transpose to get client-wise distributions
    client_distributions = []
    for client_idx in range(num_clients):
        client_dist = []
        for class_idx in range(num_classes):
            client_dist.append(round(class_distributions[class_idx][client_idx], 2))
        client_distributions.append(client_dist)
    
    return client_distributions

def normalize_distributions(client_distributions: List[List[float]]) -> List[List[float]]:
    """
    Normalize distributions to ensure each client's percentages sum to 100%
    and each class's percentages across clients sum to 100%.
    """
    num_clients = len(client_distributions)
    num_classes = len(client_distributions[0])
    
    # First, normalize each client's distribution to sum to 100%
    normalized_client_dists = []
    for client_dist in client_distributions:
        total = sum(client_dist)
        if total > 0:
            normalized_dist = [round((p / total) * 100, 2) for p in client_dist]
        else:
            normalized_dist = [round(100 / num_classes, 2)] * num_classes
        normalized_client_dists.append(normalized_dist)
    
    # Then adjust to ensure each class sums to 100% across clients
    for class_idx in range(num_classes):
        class_total = sum(client_dist[class_idx] for client_dist in normalized_client_dists)
        if class_total > 0:
            adjustment_factor = 100 / class_total
            for client_idx in range(num_clients):
                normalized_client_dists[client_idx][class_idx] = round(
                    normalized_client_dists[client_idx][class_idx] * adjustment_factor, 2
                )
    
    return normalized_client_dists

def generate_hetero_distribution(dataset_name: str, num_clients: int = 6, alpha: float = 1.0) -> Dict:
    """
    Generate heterogeneous distribution for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'bloodmnist')
        num_clients: Number of clients (default: 6)
        alpha: Dirichlet concentration parameter (default: 1.0)
    
    Returns:
        Dictionary containing client distributions and metadata
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    num_classes = config["num_classes"]
    class_names = config["class_names"]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate distributions
    client_distributions = generate_dirichlet_distribution(num_clients, num_classes, alpha)
    
    # Normalize to ensure proper constraints
    normalized_distributions = normalize_distributions(client_distributions)
    
    # Create output structure
    hetero_data = {
        "dataset": dataset_name,
        "num_clients": num_clients,
        "num_classes": num_classes,
        "alpha": alpha,
        "class_names": class_names,
        "client_distributions": {}
    }
    
    # Add client-specific distributions
    for client_idx in range(num_clients):
        client_key = f"client_{client_idx}"
        hetero_data["client_distributions"][client_key] = {
            "class_percentages": normalized_distributions[client_idx],
            "class_distribution": {
                class_names[class_idx]: normalized_distributions[client_idx][class_idx]
                for class_idx in range(num_classes)
            }
        }
    
    # Add validation info
    hetero_data["validation"] = {
        "class_totals": [
            sum(normalized_distributions[client_idx][class_idx] for client_idx in range(num_clients))
            for class_idx in range(num_classes)
        ]
    }
    
    return hetero_data

def save_hetero_distribution(dataset_name: str, output_dir: str = None, alpha: float = 1.0, num_clients: int = 6):
    """
    Generate and save heterogeneous distribution to JSON file.
    """
    if output_dir is None:
        output_dir = f"config/{dataset_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate hetero data
    hetero_data = generate_hetero_distribution(dataset_name, num_clients, alpha)
    
    # Save to file
    output_path = os.path.join(output_dir, "hetero_data.json")
    with open(output_path, 'w') as f:
        json.dump(hetero_data, f, indent=2)
    
    print(f"✅ {dataset_name}: Heterogeneous distribution saved to: {output_path}")
    
    # Print summary
    print(f"   Dataset: {hetero_data['dataset']}")
    print(f"   Clients: {hetero_data['num_clients']}")
    print(f"   Classes: {hetero_data['num_classes']}")
    print(f"   Alpha: {hetero_data['alpha']}")
    print(f"   Class names: {hetero_data['class_names'][:3]}{'...' if len(hetero_data['class_names']) > 3 else ''}")
    print()

def generate_all_datasets(alpha: float = 1.0, num_clients: int = 6):
    """
    Generate heterogeneous distributions for all supported datasets.
    """
    print("Generating Heterogeneous Dataset Distributions")
    print("=" * 60)
    print(f"Parameters: alpha={alpha}, num_clients={num_clients}")
    print()
    
    target_datasets = [
        "bloodmnist", "breastmnist", "chestmnist", "dermamnist", "octmnist",
        "organamnist", "organcmnist", "organsmnist", "pathmnist", 
        "pneumoniamnist", "retinamnist", "tissuemnist"
    ]
    
    for dataset_name in target_datasets:
        try:
            save_hetero_distribution(dataset_name, alpha=alpha, num_clients=num_clients)
        except Exception as e:
            print(f"❌ {dataset_name}: Error generating distribution: {e}")
    
    print("=" * 60)
    print(f"✅ Completed generation for {len(target_datasets)} datasets")
    
    # Generate summary
    print("\n📊 Dataset Summary:")
    for dataset_name in target_datasets:
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            print(f"  {dataset_name:15}: {config['num_classes']:2d} classes")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate heterogeneous dataset distributions")
    parser.add_argument("--dataset", type=str, help="Specific dataset to generate (optional)")
    parser.add_argument("--all", action="store_true", help="Generate for all datasets")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet alpha parameter")
    parser.add_argument("--clients", type=int, default=6, help="Number of clients")
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_datasets(alpha=args.alpha, num_clients=args.clients)
    elif args.dataset:
        if args.dataset in DATASET_CONFIGS:
            save_hetero_distribution(args.dataset, alpha=args.alpha, num_clients=args.clients)
        else:
            print(f"❌ Dataset {args.dataset} not supported.")
            print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
    else:
        print("Please specify --dataset <name> or --all")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")

if __name__ == "__main__":
    main() 