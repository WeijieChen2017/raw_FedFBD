import os
import pandas as pd
from pathlib import Path
import json
import re
from collections import defaultdict

def parse_folder_name_for_alpha(folder_name):
    """
    Parse folder name divided by underscores to extract attributes, focusing on alpha.
    """
    parts = folder_name.split('_')
    attributes = {}
    attributes['full_folder_name'] = folder_name
    
    # Look for alpha value - check various patterns
    alpha_value = None
    
    # Pattern 1: alpha followed by underscore and number
    for i, part in enumerate(parts):
        if 'alpha' in part.lower() and i + 1 < len(parts):
            try:
                alpha_value = float(parts[i + 1])
                break
            except:
                pass
    
    # Pattern 2: alpha with number in same part (alpha0.25)
    if alpha_value is None:
        for part in parts:
            match = re.search(r'alpha[_\-=]?(\d+\.?\d*)', part, re.IGNORECASE)
            if match:
                try:
                    alpha_value = float(match.group(1))
                    # If it's a large number like 25, might mean 0.25
                    if alpha_value > 1:
                        alpha_value = alpha_value / 100
                    break
                except:
                    pass
    
    if alpha_value is not None:
        attributes['alpha'] = alpha_value
    
    # Extract other common attributes
    medical_datasets = ['bloodmnist', 'organamnist', 'pathmnist', 'dermamnist', 
                      'octmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist',
                      'organsmnist', 'organcmnist', 'breastmnist', 'chestmnist', 'siim']
    
    for part in parts:
        if any(dataset in part.lower() for dataset in medical_datasets):
            attributes['dataset'] = part
            break
    
    model_names = ['resnet18', 'resnet50', 'vgg16', 'densenet121', 'unet', 'mobilenet']
    
    for part in parts:
        if any(model in part.lower() for model in model_names):
            attributes['model'] = part
            break
    
    # Date pattern (YYYYMMDD)
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            attributes['date'] = part
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                attributes['time'] = parts[i + 1]
    
    return attributes

def read_result_files(folder_path):
    """
    Read result files from the folder and extract metrics.
    """
    results = {}
    
    # Common result file patterns
    result_files = [
        'eval_metrics.json',
        'results.json',
        'test_results.json',
        'final_results.json',
        'server_evaluation_history.json'
    ]
    
    # Try to find and read JSON result files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json') and any(pattern in file.lower() for pattern in ['eval', 'result', 'metric', 'test']):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Extract metrics based on common patterns
                        if isinstance(data, dict):
                            # Look for accuracy, loss, etc.
                            for key in ['accuracy', 'acc', 'test_accuracy', 'val_accuracy', 
                                       'loss', 'test_loss', 'val_loss', 'f1', 'auc', 'dice']:
                                if key in data:
                                    results[f'result_{key}'] = data[key]
                            
                            # If data has nested structure (e.g., per epoch/round)
                            if 'rounds' in data or 'epochs' in data:
                                rounds_data = data.get('rounds', data.get('epochs', []))
                                if rounds_data and isinstance(rounds_data, list):
                                    # Get last round/epoch results
                                    last_round = rounds_data[-1]
                                    if isinstance(last_round, dict):
                                        for key, value in last_round.items():
                                            if any(metric in key.lower() for metric in ['acc', 'loss', 'f1', 'auc']):
                                                results[f'final_{key}'] = value
                        
                        # Break after finding first valid result file
                        if results:
                            results['result_file'] = file
                            break
                except:
                    pass
    
    # Try CSV files if no JSON results found
    if not results:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv') and any(pattern in file.lower() for pattern in ['eval', 'result', 'metric']):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        # Get last row of results
                        if not df.empty:
                            last_row = df.iloc[-1]
                            for col in df.columns:
                                if any(metric in col.lower() for metric in ['acc', 'loss', 'f1', 'auc']):
                                    results[f'result_{col}'] = last_row[col]
                            results['result_file'] = file
                            break
                    except:
                        pass
    
    return results

def scan_and_process_fbd_run(fbd_run_path):
    """
    Scan the fbd_run directory, filter out tau folders, extract attributes and results.
    """
    all_folders_data = []
    tau_count = 0
    
    if not os.path.exists(fbd_run_path):
        print(f"Error: Directory {fbd_run_path} does not exist!")
        return all_folders_data, tau_count
    
    folders = [f for f in os.listdir(fbd_run_path) if os.path.isdir(os.path.join(fbd_run_path, f))]
    total_folders = len(folders)
    
    print(f"Found {total_folders} folders to process...")
    
    for i, item in enumerate(folders):
        item_path = os.path.join(fbd_run_path, item)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processing folder {i + 1}/{total_folders}...")
        
        # Check if folder contains 'tau'
        if 'tau' in item.lower():
            tau_count += 1
            continue  # Skip tau folders
        
        # Parse the folder name
        attributes = parse_folder_name_for_alpha(item)
        
        # Read result files
        results = read_result_files(item_path)
        
        # Merge results with attributes
        attributes.update(results)
        
        # Check if has any results
        attributes['has_results'] = len(results) > 0
        
        all_folders_data.append(attributes)
    
    return all_folders_data, tau_count

def create_comprehensive_excel(folders_data, output_path='fbd_alpha_results_with_data.xlsx'):
    """
    Create Excel file with alpha-focused analysis including actual results.
    """
    if not folders_data:
        print("No folders found to process!")
        return
    
    df = pd.DataFrame(folders_data)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Identify result columns
        result_cols = [col for col in df.columns if col.startswith('result_') or col.startswith('final_')]
        
        # Sheet 1: All non-tau results
        priority_cols = ['alpha', 'full_folder_name', 'dataset', 'model', 'has_results']
        priority_cols.extend(result_cols)
        priority_cols.extend(['date', 'time'])
        
        other_cols = [col for col in df.columns if col not in priority_cols]
        ordered_cols = [col for col in priority_cols if col in df.columns] + other_cols
        
        df_ordered = df[ordered_cols]
        df_ordered.to_excel(writer, sheet_name='All_Non_Tau_Results', index=False)
        
        # Sheet 2: Only folders with alpha values
        if 'alpha' in df.columns:
            df_with_alpha = df[df['alpha'].notna()].copy()
            if not df_with_alpha.empty:
                df_with_alpha = df_with_alpha[ordered_cols]
                df_with_alpha.to_excel(writer, sheet_name='Alpha_Folders_Only', index=False)
                
                # Sheets grouped by alpha value
                for alpha_value in sorted(df_with_alpha['alpha'].unique()):
                    alpha_df = df_with_alpha[df_with_alpha['alpha'] == alpha_value]
                    sheet_name = f'Alpha_{str(alpha_value).replace(".", "_")}'[:31]
                    alpha_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Sheet 3: Summary of results by alpha
        if 'alpha' in df.columns and result_cols:
            summary_data = []
            for alpha_value in sorted(df['alpha'].dropna().unique()):
                alpha_df = df[df['alpha'] == alpha_value]
                summary_row = {'alpha': alpha_value, 'experiment_count': len(alpha_df)}
                
                # Calculate mean of numeric result columns
                for col in result_cols:
                    if col in alpha_df.columns:
                        numeric_data = pd.to_numeric(alpha_df[col], errors='coerce')
                        if numeric_data.notna().any():
                            summary_row[f'mean_{col}'] = numeric_data.mean()
                            summary_row[f'std_{col}'] = numeric_data.std()
                
                summary_data.append(summary_row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Alpha_Results_Summary', index=False)
        
        # Auto-adjust column widths for all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    print(f"Comprehensive Excel file created: {output_path}")

def main():
    """
    Main function to aggregate results focusing on alpha attribute with actual results.
    """
    fbd_run_path = './fbd_run'
    
    print(f"Scanning directory: {fbd_run_path}")
    print("Filtering out folders containing 'tau'...")
    print("Extracting 'alpha' values and reading result files...")
    
    # Scan directory and extract attributes with results
    folders_data, tau_count = scan_and_process_fbd_run(fbd_run_path)
    
    if not folders_data:
        print("No non-tau folders found in the specified directory!")
        return
    
    print(f"\nProcessing complete:")
    print(f"- Total tau folders filtered out: {tau_count}")
    print(f"- Non-tau folders processed: {len(folders_data)}")
    
    # Count various statistics
    df_temp = pd.DataFrame(folders_data)
    if 'alpha' in df_temp.columns:
        alpha_count = df_temp['alpha'].notna().sum()
        print(f"- Folders with alpha values: {alpha_count}")
        if alpha_count > 0:
            print(f"- Unique alpha values: {sorted(df_temp['alpha'].dropna().unique())}")
    
    folders_with_results = df_temp['has_results'].sum() if 'has_results' in df_temp.columns else 0
    print(f"- Folders with result files: {folders_with_results}")
    
    # Create comprehensive Excel file
    create_comprehensive_excel(folders_data)
    
    print("\nFile created: fbd_alpha_results_with_data.xlsx")
    print("\nThis file includes:")
    print("- All non-tau folders with their attributes")
    print("- Actual results extracted from JSON/CSV files in each folder")
    print("- Sheets grouped by alpha values")
    print("- Summary statistics of results by alpha value")

if __name__ == "__main__":
    main()