import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
import medmnist
from medmnist import INFO
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for high-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

class TeeOutput:
    """Class to duplicate output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

def class_separability_report(X, y, dataset_name, split_name, output_dir="class_separability_reports"):
    """
    Generate comprehensive class separability report for a dataset.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        dataset_name: Name of the dataset
        output_dir: Directory to save outputs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Always standardize before distance-based analytics
    X_scaled = StandardScaler().fit_transform(X)
    
    # Calculate intra-class distances
    def intra_class_distances(X, y, metric="euclidean"):
        d_intra = {}
        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) > 1:  # Need at least 2 samples for distance calculation
                dists = pdist(Xc, metric=metric)
                d_intra[c] = dists.mean()
            else:
                d_intra[c] = 0.0
        return d_intra
    
    intra = intra_class_distances(X_scaled, y)
    
    # Calculate inter-class distances (centroid distances)
    classes = np.unique(y)
    centroids = np.vstack([X_scaled[y == c].mean(0) for c in classes])
    inter_distances = cdist(centroids, centroids, metric="euclidean")
    
    # Calculate clustering metrics
    sil = silhouette_score(X_scaled, y)
    ch = calinski_harabasz_score(X_scaled, y)
    db = davies_bouldin_score(X_scaled, y)
    
    # Print report
    print(f"\n{'='*60}")
    print(f"CLASS SEPARABILITY REPORT: {dataset_name.upper()} - {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split_name}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    print(f"\n=== INTRA-CLASS MEAN DISTANCES ===")
    for c, d in intra.items():
        class_size = len(y[y == c])
        print(f"Class {c}: {d:.4f} (n={class_size})")
    
    print(f"\n=== CLUSTERING METRICS ===")
    print(f"Silhouette Score      : {sil:.4f} (range: [-1, 1], higher is better)")
    print(f"Calinski-Harabasz (CH): {ch:.2f} (higher is better)")
    print(f"Davies-Bouldin (DB)   : {db:.4f} (lower is better)")
    
    # Create and save heatmap
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(inter_distances, dtype=bool), k=1)
    sns.heatmap(inter_distances, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap="viridis",
                xticklabels=classes, 
                yticklabels=classes,
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title(f"Inter-Class Centroid Distances\n{dataset_name} - {split_name}")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(output_dir, f"{dataset_name}_{split_name}_class_separability.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {figure_path}")
    plt.close()
    
    # Save numerical results
    results = {
        'dataset': dataset_name,
        'split': split_name,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(classes),
        'classes': classes.tolist(),
        'intra_class_distances': intra,
        'silhouette_score': sil,
        'calinski_harabasz_score': ch,
        'davies_bouldin_score': db,
        'inter_class_distances': inter_distances.tolist()
    }
    
    return results

def extract_features_from_dataset(dataset, max_samples=5000):
    """
    Extract features from a MEDMNIST dataset by flattening images.
    
    Args:
        dataset: MEDMNIST dataset object
        max_samples: Maximum number of samples to use (for computational efficiency)
    
    Returns:
        X: Feature matrix (flattened images)
        y: Labels
    """
    # Limit samples for computational efficiency
    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    X = []
    y = []
    
    for idx in indices:
        img, label = dataset[idx]
        
        # Convert PIL Image to numpy array
        if hasattr(img, 'mode'):  # PIL Image
            img = np.array(img)
        elif hasattr(img, 'numpy'):  # PyTorch tensor
            img = img.numpy()
        elif not isinstance(img, np.ndarray):  # Other formats
            img = np.array(img)
        
        # Ensure image is in proper format and flatten
        if img.ndim == 3 and img.shape[2] == 1:  # (H, W, 1) -> (H, W)
            img = img.squeeze(axis=2)
        elif img.ndim == 3 and img.shape[0] == 1:  # (1, H, W) -> (H, W)
            img = img.squeeze(axis=0)
        elif img.ndim == 3 and img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            img = img.transpose(1, 2, 0)
        
        # Flatten the image
        img_flat = img.flatten()
        X.append(img_flat)
        
        # Handle label format
        if hasattr(label, 'item'):
            try:
                label = label.item()
            except ValueError:
                # Multi-dimensional label, convert to single class
                label = np.array(label).flatten()
                if len(label) == 1:
                    label = label[0]
                else:
                    # For multi-label, use first positive label or create composite label
                    if np.any(label == 1):
                        label = np.where(label == 1)[0][0]  # First positive class
                    else:
                        label = int(np.sum(label * np.arange(len(label))))  # Weighted sum
        elif isinstance(label, np.ndarray):
            if label.size == 1:
                label = label.item()
            else:
                # Multi-label case
                label = label.flatten()
                if np.any(label == 1):
                    label = np.where(label == 1)[0][0]  # First positive class
                else:
                    label = int(np.sum(label * np.arange(len(label))))  # Weighted sum
        elif hasattr(label, '__len__') and len(label) == 1:
            label = label[0]
        elif hasattr(label, '__len__') and len(label) > 1:
            # Multi-label case
            label = np.array(label)
            if np.any(label == 1):
                label = np.where(label == 1)[0][0]  # First positive class
            else:
                label = int(np.sum(label * np.arange(len(label))))  # Weighted sum
        
        y.append(int(label))
    
    return np.array(X), np.array(y)

def main():
    """Generate class separability reports for 12 MEDMNIST 2D datasets across train/val/test splits."""
    
    # Create output directory
    output_dir = "class_separability_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup output logging to file
    log_file_path = os.path.join(output_dir, "class_separability_analysis_log.txt")
    tee_output = TeeOutput(log_file_path)
    sys.stdout = tee_output
    
    print("="*80)
    print("MEDMNIST CLASS SEPARABILITY ANALYSIS")
    print("="*80)
    print(f"Log file: {log_file_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Get all 2D datasets (exclude 3D datasets)
    all_2d_datasets = []
    for flag, info in INFO.items():
        if '3d' not in flag.lower():
            all_2d_datasets.append(flag)
    
    print(f"Found {len(all_2d_datasets)} 2D datasets: {all_2d_datasets}")
    
    # Select 12 datasets (use all if there are exactly 12, otherwise select first 12)
    target_datasets = all_2d_datasets[:12]
    
    print(f"\nGenerating class separability reports for {len(target_datasets)} datasets:")
    print(f"Target datasets: {target_datasets}")
    
    # Store all results for summary
    all_results = []
    splits = ['train', 'val', 'test']
    
    # Process each dataset
    for i, data_flag in enumerate(target_datasets, 1):
        print(f"\n[{i}/{len(target_datasets)}] Processing {data_flag}...")
        
        try:
            # Get dataset info and class
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            
            # Process each split
            for split in splits:
                print(f"  Processing {split} split...")
                
                try:
                    # Load dataset split
                    dataset = DataClass(split=split, download=True, root='./cache/')
                    
                    # Extract features and labels
                    print(f"    Extracting features from {len(dataset)} samples...")
                    X, y = extract_features_from_dataset(dataset, max_samples=5000)
                    
                    print(f"    Extracted features: {X.shape}, Labels: {y.shape}")
                    print(f"    Unique classes: {np.unique(y)}")
                    
                    # Generate separability report
                    results = class_separability_report(X, y, data_flag, split, output_dir)
                    all_results.append(results)
                    
                    print(f"    ✅ Completed {data_flag} - {split}")
                    
                except Exception as e:
                    print(f"    ❌ Error processing {data_flag} - {split}: {str(e)}")
                    continue
            
            print(f"✅ Completed all splits for {data_flag}")
            
        except Exception as e:
            print(f"❌ Error processing {data_flag}: {str(e)}")
            continue
    
    # Generate summary report and save as Excel
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL DATASETS")
    print(f"{'='*60}")
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
            'Split': result['split'],
            'Classes': result['n_classes'],
            'Samples': result['n_samples'],
            'Features': result['n_features'],
            'Silhouette': result['silhouette_score'],
            'Calinski-Harabasz': result['calinski_harabasz_score'],
            'Davies-Bouldin': result['davies_bouldin_score']
        })
    
    # Print and save summary
    if summary_data:
        import pandas as pd
        df = pd.DataFrame(summary_data)
        
        # Format for display
        df_display = df.copy()
        df_display['Silhouette'] = df_display['Silhouette'].apply(lambda x: f"{x:.4f}")
        df_display['Calinski-Harabasz'] = df_display['Calinski-Harabasz'].apply(lambda x: f"{x:.2f}")
        df_display['Davies-Bouldin'] = df_display['Davies-Bouldin'].apply(lambda x: f"{x:.4f}")
        
        print(df_display.to_string(index=False))
        
        # Save summary as Excel with multiple sheets
        excel_path = os.path.join(output_dir, "class_separability_summary.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main summary sheet
            df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create separate sheets for each dataset
            for dataset in target_datasets:
                dataset_data = df[df['Dataset'] == dataset]
                if not dataset_data.empty:
                    dataset_data.to_excel(writer, sheet_name=dataset, index=False)
            
            # Create separate sheets for each split
            for split in splits:
                split_data = df[df['Split'] == split]
                if not split_data.empty:
                    split_data.to_excel(writer, sheet_name=f'All_{split}', index=False)
        
        print(f"\nSaved Excel summary: {excel_path}")
        
        # Also save individual results as Excel files
        detailed_results = []
        for result in all_results:
            # Flatten intra-class distances for Excel
            row = {
                'Dataset': result['dataset'],
                'Split': result['split'],
                'n_samples': result['n_samples'],
                'n_features': result['n_features'],
                'n_classes': result['n_classes'],
                'silhouette_score': result['silhouette_score'],
                'calinski_harabasz_score': result['calinski_harabasz_score'],
                'davies_bouldin_score': result['davies_bouldin_score']
            }
            
            # Add intra-class distances as separate columns
            for class_id, distance in result['intra_class_distances'].items():
                row[f'intra_class_dist_{class_id}'] = distance
            
            detailed_results.append(row)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_excel_path = os.path.join(output_dir, "class_separability_detailed.xlsx")
        detailed_df.to_excel(detailed_excel_path, index=False)
        print(f"Saved detailed Excel results: {detailed_excel_path}")
    
    print(f"\n✅ Class separability analysis completed!")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"📊 Generated {len(all_results)} reports with high-quality figures (300 DPI)")
    print(f"📋 Summary saved as Excel files with multiple sheets")
    print(f"🔢 Processed {len(target_datasets)} datasets across {len(splits)} splits each")
    print(f"📝 Complete log saved to: {log_file_path}")
    print("="*80)
    
    # Close the log file and restore normal output
    tee_output.close()

if __name__ == "__main__":
    main() 