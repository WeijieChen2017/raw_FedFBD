#!/usr/bin/env python3
"""
Heterogeneous Dataset Distribution Generator for OrganAMNIST
Generates client-specific class distributions for federated learning with data heterogeneity.
"""

import json
import numpy as np
import random
from typing import Dict, List

def generate_dirichlet_distribution(num_clients: int, num_classes: int, alpha: float = 0.5) -> List[List[float]]:
    """
    Generate heterogeneous data distribution using Dirichlet distribution.
    
    Args:
        num_clients: Number of clients (6)
        num_classes: Number of classes (11 for OrganAMNIST)
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
    
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

def generate_hetero_distribution(num_clients: int = 6, num_classes: int = 11, alpha: float = 0.5) -> Dict:
    """
    Generate heterogeneous distribution for OrganAMNIST dataset.
    
    Args:
        num_clients: Number of clients (default: 6)
        num_classes: Number of classes (default: 11 for OrganAMNIST)
        alpha: Dirichlet concentration parameter (default: 0.5)
    
    Returns:
        Dictionary containing client distributions and metadata
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate distributions
    client_distributions = generate_dirichlet_distribution(num_clients, num_classes, alpha)
    
    # Normalize to ensure proper constraints
    normalized_distributions = normalize_distributions(client_distributions)
    
    # Create output structure
    hetero_data = {
        "dataset": "organamnist",
        "num_clients": num_clients,
        "num_classes": num_classes,
        "alpha": alpha,
        "class_names": [
            "bladder", "femur-left", "femur-right", "heart", "kidney-left", 
            "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"
        ],
        "client_distributions": {}
    }
    
    # Add client-specific distributions
    for client_idx in range(num_clients):
        client_key = f"client_{client_idx}"
        hetero_data["client_distributions"][client_key] = {
            "class_percentages": normalized_distributions[client_idx],
            "class_distribution": {
                hetero_data["class_names"][class_idx]: normalized_distributions[client_idx][class_idx]
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

def save_hetero_distribution(output_path: str = "config/organamnist_tau/hetero_data.json"):
    """
    Generate and save heterogeneous distribution to JSON file.
    """
    hetero_data = generate_hetero_distribution()
    
    with open(output_path, 'w') as f:
        json.dump(hetero_data, f, indent=2)
    
    print(f"Heterogeneous distribution saved to: {output_path}")
    
    # Print summary
    print("\n=== Distribution Summary ===")
    print(f"Dataset: {hetero_data['dataset']}")
    print(f"Clients: {hetero_data['num_clients']}")
    print(f"Classes: {hetero_data['num_classes']}")
    print(f"Alpha: {hetero_data['alpha']}")
    
    print("\n=== Client Distributions ===")
    for client_key, client_data in hetero_data["client_distributions"].items():
        print(f"{client_key}: {client_data['class_percentages']}")
    
    print("\n=== Validation ===")
    print(f"Class totals: {hetero_data['validation']['class_totals']}")

if __name__ == "__main__":
    save_hetero_distribution()