#!/usr/bin/env python3
"""
Test script to generate and analyze heterogeneous dataset distribution for OrganAMNIST
"""

import sys
import json
import pandas as pd
import numpy as np
import torch
from collections import Counter, defaultdict
from fbd_main_tau import load_hetero_config, create_hetero_partitions

class MockDataset:
    """Mock dataset that simulates OrganAMNIST structure"""
    def __init__(self, samples_per_class=1000):
        self.samples_per_class = samples_per_class
        self.num_classes = 11
        self.total_samples = samples_per_class * self.num_classes
        
        # Create balanced dataset with equal samples per class
        self.labels = []
        self.data = []
        
        for class_idx in range(self.num_classes):
            for _ in range(samples_per_class):
                self.labels.append(class_idx)
                self.data.append(np.random.randn(28, 28))  # Mock image data
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def analyze_client_distributions(partitions, hetero_config):
    """
    Analyze the actual distribution of samples across clients
    
    Args:
        partitions: List of client partitions (Subset objects)
        hetero_config: Heterogeneous configuration
    
    Returns:
        dict: Analysis results
    """
    num_clients = len(partitions)
    num_classes = hetero_config['num_classes']
    class_names = hetero_config['class_names']
    
    # Initialize results
    results = {
        'client_stats': [],
        'class_distribution': defaultdict(list),
        'summary': {}
    }
    
    total_samples = 0
    overall_class_counts = Counter()
    
    # Analyze each client
    for client_idx, partition in enumerate(partitions):
        client_samples = len(partition)
        total_samples += client_samples
        
        # Count classes in this client's partition
        client_class_counts = Counter()
        for i in range(client_samples):
            _, label = partition[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            client_class_counts[label] += 1
            overall_class_counts[label] += 1
        
        # Calculate percentages
        client_percentages = []
        for class_idx in range(num_classes):
            count = client_class_counts[class_idx]
            percentage = (count / client_samples * 100) if client_samples > 0 else 0
            client_percentages.append(percentage)
            
            # Store for class distribution analysis
            results['class_distribution'][class_idx].append(count)
        
        # Store client stats
        client_stats = {
            'client_id': client_idx,
            'total_samples': client_samples,
            'class_counts': [client_class_counts[i] for i in range(num_classes)],
            'class_percentages': client_percentages,
            'expected_percentages': hetero_config['client_distributions'][f'client_{client_idx}']['class_percentages']
        }
        results['client_stats'].append(client_stats)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_samples': total_samples,
        'num_clients': num_clients,
        'num_classes': num_classes,
        'class_names': class_names,
        'overall_class_counts': [overall_class_counts[i] for i in range(num_classes)],
        'samples_per_client': [len(p) for p in partitions]
    }
    
    return results

def generate_excel_report(results, output_path):
    """
    Generate Excel report with multiple sheets for distribution analysis
    
    Args:
        results: Analysis results from analyze_client_distributions
        output_path: Path to save Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Client Distribution Summary
        client_data = []
        for stats in results['client_stats']:
            row = {
                'Client ID': stats['client_id'],
                'Total Samples': stats['total_samples'],
            }
            
            # Add class counts
            for i, class_name in enumerate(results['summary']['class_names']):
                row[f'{class_name}_count'] = stats['class_counts'][i]
            
            # Add class percentages
            for i, class_name in enumerate(results['summary']['class_names']):
                row[f'{class_name}_percentage'] = round(stats['class_percentages'][i], 2)
            
            client_data.append(row)
        
        client_df = pd.DataFrame(client_data)
        client_df.to_excel(writer, sheet_name='Client_Distribution', index=False)
        
        # Sheet 2: Expected vs Actual Percentages
        comparison_data = []
        for stats in results['client_stats']:
            for class_idx, class_name in enumerate(results['summary']['class_names']):
                comparison_data.append({
                    'Client ID': stats['client_id'],
                    'Class': class_name,
                    'Expected Percentage': round(stats['expected_percentages'][class_idx], 2),
                    'Actual Percentage': round(stats['class_percentages'][class_idx], 2),
                    'Difference': round(stats['class_percentages'][class_idx] - stats['expected_percentages'][class_idx], 2),
                    'Sample Count': stats['class_counts'][class_idx]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Expected_vs_Actual', index=False)
        
        # Sheet 3: Class Distribution Across Clients
        class_dist_data = []
        for class_idx, class_name in enumerate(results['summary']['class_names']):
            row = {'Class': class_name}
            total_class_samples = results['summary']['overall_class_counts'][class_idx]
            
            for client_idx in range(results['summary']['num_clients']):
                client_samples = results['class_distribution'][class_idx][client_idx]
                percentage = (client_samples / total_class_samples * 100) if total_class_samples > 0 else 0
                row[f'Client_{client_idx}_samples'] = client_samples
                row[f'Client_{client_idx}_percentage'] = round(percentage, 2)
            
            row['Total_samples'] = total_class_samples
            class_dist_data.append(row)
        
        class_dist_df = pd.DataFrame(class_dist_data)
        class_dist_df.to_excel(writer, sheet_name='Class_Distribution', index=False)
        
        # Sheet 4: Summary Statistics
        summary_data = [
            {'Metric': 'Total Samples', 'Value': results['summary']['total_samples']},
            {'Metric': 'Number of Clients', 'Value': results['summary']['num_clients']},
            {'Metric': 'Number of Classes', 'Value': results['summary']['num_classes']},
            {'Metric': 'Average Samples per Client', 'Value': round(results['summary']['total_samples'] / results['summary']['num_clients'], 2)},
        ]
        
        # Add per-client sample counts
        for i, count in enumerate(results['summary']['samples_per_client']):
            summary_data.append({'Metric': f'Client {i} Samples', 'Value': count})
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

def main():
    """Main function to run the distribution analysis"""
    print("Heterogeneous Dataset Distribution Analysis")
    print("=" * 50)
    
    # Load heterogeneous configuration
    hetero_config = load_hetero_config('organamnist_tau')
    if hetero_config is None:
        print("❌ Failed to load hetero config")
        return False
    
    print("✅ Loaded hetero config successfully")
    print(f"   Dataset: {hetero_config['dataset']}")
    print(f"   Clients: {hetero_config['num_clients']}")
    print(f"   Classes: {hetero_config['num_classes']}")
    print(f"   Alpha: {hetero_config['alpha']}")
    
    # Create mock dataset (1000 samples per class = 11,000 total)
    mock_dataset = MockDataset(samples_per_class=1000)
    print(f"✅ Created mock dataset with {len(mock_dataset)} samples")
    
    # Create heterogeneous partitions
    try:
        partitions = create_hetero_partitions(mock_dataset, hetero_config, seed=42)
        print(f"✅ Created {len(partitions)} client partitions")
    except Exception as e:
        print(f"❌ Failed to create partitions: {e}")
        return False
    
    # Analyze distributions
    print("\nAnalyzing client distributions...")
    results = analyze_client_distributions(partitions, hetero_config)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS RESULTS")
    print("=" * 60)
    
    for stats in results['client_stats']:
        print(f"\nClient {stats['client_id']}:")
        print(f"  Total samples: {stats['total_samples']}")
        print("  Class distribution:")
        for i, class_name in enumerate(results['summary']['class_names']):
            count = stats['class_counts'][i]
            percentage = stats['class_percentages'][i]
            expected = stats['expected_percentages'][i]
            print(f"    {class_name:12}: {count:4d} samples ({percentage:5.1f}%) [expected: {expected:5.1f}%]")
    
    # Generate Excel report
    output_path = "heterogeneous_distribution_analysis_tau.xlsx"
    try:
        generate_excel_report(results, output_path)
        print(f"\n✅ Excel report generated: {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate Excel report: {e}")
        return False
    
    print("\n✅ Analysis completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)