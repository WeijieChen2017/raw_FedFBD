folder = "col2_vanilla_2d"
# folder = "col1_eval_2d"

DATASETS_2D = [
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
]

import glob
import json
import pandas as pd

# Find all *_eval.json files recursively in col2_vanilla_2d
result_json_list = sorted(glob.glob(f"{folder}/**/*_eval.json", recursive=True))
print("All evaluation JSON files found in col2_vanilla_2d:")
print("=" * 60)
for json_path in result_json_list:
    print(json_path)

print(f"\nTotal files found: {len(result_json_list)}")

# Extract metrics from all JSON files
data_rows = []

print("\nExtracting metrics from JSON files...")
print("=" * 60)

for json_path in result_json_list:
    try:
        # Parse path components
        dataset = json_path.split("/")[1]
        model = json_path.split("/")[2]
        image_size = json_path.split("/")[3]
        model_index = json_path.split("/")[4]
        
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract metrics
        row = {
            'dataset': dataset,
            'model': model,
            'image_size': image_size,
            'model_index': model_index,
            'train_auc': data.get('train_auc', None),
            'train_acc': data.get('train_acc', None),
            'val_auc': data.get('val_auc', None),
            'val_acc': data.get('val_acc', None),
            'test_auc': data.get('test_auc', None),
            'test_acc': data.get('test_acc', None),
            'train_loss': data.get('train_loss', None),
            'val_loss': data.get('val_loss', None),
            'test_loss': data.get('test_loss', None),
            'num_epochs': data.get('num_epochs', None),
            'batch_size': data.get('batch_size', None),
            'lr': data.get('lr', None),
            'file_path': json_path
        }
        data_rows.append(row)
        print(f"✓ {json_path} -> Dataset: {dataset}, Model: {model}, Test AUC: {data.get('test_auc', 'N/A'):.4f}, Test ACC: {data.get('test_acc', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"✗ Error reading {json_path}: {e}")

# Create DataFrame and save to CSV
if data_rows:
    df = pd.DataFrame(data_rows)
    
    # Sort by dataset, model, and model_index for better organization
    df = df.sort_values(['dataset', 'model', 'model_index'])
    
    # Save to CSV
    output_file = f"{folder}_metrics.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully extracted metrics from {len(data_rows)} files")
    print(f"✓ Results saved to: {output_file}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"- Datasets: {df['dataset'].nunique()} unique ({', '.join(sorted(df['dataset'].unique()))})")
    print(f"- Models: {df['model'].nunique()} unique ({', '.join(sorted(df['model'].unique()))})")
    print(f"- Average Test AUC: {df['test_auc'].mean():.4f} (±{df['test_auc'].std():.4f})")
    print(f"- Average Test ACC: {df['test_acc'].mean():.4f} (±{df['test_acc'].std():.4f})")
    
else:
    print("No data extracted!")

# col2_vanilla_2d/bloodmnist/resnet18/28/1/resnet18_28_1_eval.json