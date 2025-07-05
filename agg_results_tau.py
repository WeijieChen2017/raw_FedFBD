import os
import json
import pandas as pd
from pathlib import Path
import glob

def find_tau_folders(base_dir="fbd_run"):
    """Find all folders containing 'tau' in the fbd_run directory."""
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []
    
    tau_folders = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "_tau" in dir_name.lower():
                tau_folders.append(os.path.join(root, dir_name))
    
    return tau_folders

def extract_evaluation_metrics(folder_path):
    """Extract evaluation metrics from the last item in server_evaluation_history.json."""
    eval_file = os.path.join(folder_path, "eval_results", "server_evaluation_history.json")
    
    if not os.path.exists(eval_file):
        print(f"Evaluation file not found: {eval_file}")
        return None
    
    try:
        with open(eval_file, 'r') as f:
            history = json.load(f)
        
        if not history or not isinstance(history, list):
            print(f"Invalid history format in {eval_file}")
            return None
        
        # Get the last item
        last_item = history[-1]
        
        # Extract metrics from averaging and ensemble
        results = {
            'folder_name': os.path.basename(folder_path),
            'averaging_test_auc': None,
            'averaging_test_acc': None,
            'ensemble_test_auc': None,
            'ensemble_test_acc': None
        }
        
        if 'averaging' in last_item and isinstance(last_item['averaging'], dict):
            results['averaging_test_auc'] = last_item['averaging'].get('test_auc')
            results['averaging_test_acc'] = last_item['averaging'].get('test_acc')
        
        if 'ensemble' in last_item and isinstance(last_item['ensemble'], dict):
            results['ensemble_test_auc'] = last_item['ensemble'].get('test_auc')
            results['ensemble_test_acc'] = last_item['ensemble'].get('test_acc')
        
        return results
        
    except Exception as e:
        print(f"Error reading {eval_file}: {e}")
        return None

def main():
    """Main function to aggregate tau results and create Excel table."""
    print("Searching for tau folders...")
    
    # Find all tau folders
    tau_folders = find_tau_folders()
    
    if not tau_folders:
        print("No tau folders found in fbd_run directory")
        return
    
    print(f"Found {len(tau_folders)} tau folders:")
    for folder in tau_folders:
        print(f"  - {folder}")
    
    # Extract metrics from each folder
    results = []
    for folder in tau_folders:
        print(f"\nProcessing: {folder}")
        metrics = extract_evaluation_metrics(folder)
        if metrics:
            results.append(metrics)
            print(f"  ✓ Extracted metrics for {metrics['folder_name']}")
        else:
            print(f"  ✗ Failed to extract metrics")
    
    if not results:
        print("No valid results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'folder_name',
        'averaging_test_auc',
        'averaging_test_acc', 
        'ensemble_test_auc',
        'ensemble_test_acc'
    ]
    df = df[column_order]
    
    # Save to Excel
    output_file = "tau_results.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total folders processed: {len(results)}")
    
    # Display summary
    print("\nSummary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
