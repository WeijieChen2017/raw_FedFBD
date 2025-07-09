import os
import pandas as pd
from pathlib import Path
import json
import re
from collections import defaultdict

def parse_folder_name_for_alpha(folder_name):
    """
    Parse folder name divided by underscores to extract attributes, focusing on alpha.
    Alpha might appear in patterns like:
    - alpha_0.25
    - alpha0.25
    - alpha=0.25
    - a_0.25
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
    # Dataset names
    medical_datasets = ['bloodmnist', 'organamnist', 'pathmnist', 'dermamnist', 
                      'octmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist',
                      'organsmnist', 'organcmnist', 'breastmnist', 'chestmnist', 'siim']
    
    for part in parts:
        if any(dataset in part.lower() for dataset in medical_datasets):
            attributes['dataset'] = part
            break
    
    # Model names
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
    
    # Store all parts for debugging
    for i, part in enumerate(parts):
        attributes[f'part_{i}'] = part
    
    return attributes

def scan_and_filter_fbd_run(fbd_run_path):
    """
    Scan the fbd_run directory, filter out tau folders, and extract attributes.
    """
    all_folders_data = []
    tau_count = 0
    
    if not os.path.exists(fbd_run_path):
        print(f"Error: Directory {fbd_run_path} does not exist!")
        return all_folders_data, tau_count
    
    for item in os.listdir(fbd_run_path):
        item_path = os.path.join(fbd_run_path, item)
        if os.path.isdir(item_path):
            # Check if folder contains 'tau'
            if 'tau' in item.lower():
                tau_count += 1
                continue  # Skip tau folders
            
            # Parse the folder name
            attributes = parse_folder_name_for_alpha(item)
            
            # Check for result files
            attributes['has_results'] = check_for_results(item_path)
            
            all_folders_data.append(attributes)
    
    return all_folders_data, tau_count

def check_for_results(folder_path):
    """
    Check if the folder contains result files.
    """
    result_patterns = ['*.json', '*.csv', '*.log', 'eval_metrics.json', 'results.txt']
    
    for pattern in result_patterns:
        files = list(Path(folder_path).rglob(pattern))
        if files:
            return True
    return False

def create_alpha_focused_excel(folders_data, output_path='fbd_alpha_focused_results.xlsx'):
    """
    Create Excel file with alpha-focused analysis.
    """
    if not folders_data:
        print("No folders found to process!")
        return
    
    df = pd.DataFrame(folders_data)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All non-tau results
        priority_cols = ['alpha', 'full_folder_name', 'dataset', 'model', 'date', 'time', 'has_results']
        part_cols = [col for col in df.columns if col.startswith('part_')]
        other_cols = [col for col in df.columns if col not in priority_cols and col not in part_cols]
        ordered_cols = [col for col in priority_cols if col in df.columns] + other_cols + part_cols
        
        df_ordered = df[ordered_cols]
        df_ordered.to_excel(writer, sheet_name='All_Non_Tau_Results', index=False)
        
        # Sheet 2: Only folders with alpha values
        if 'alpha' in df.columns:
            df_with_alpha = df[df['alpha'].notna()].copy()
            if not df_with_alpha.empty:
                df_with_alpha = df_with_alpha[ordered_cols]
                df_with_alpha.to_excel(writer, sheet_name='Alpha_Folders_Only', index=False)
                
                # Sheet 3: Grouped by alpha value
                for alpha_value in sorted(df_with_alpha['alpha'].unique()):
                    alpha_df = df_with_alpha[df_with_alpha['alpha'] == alpha_value]
                    sheet_name = f'Alpha_{str(alpha_value).replace(".", "_")}'[:31]
                    alpha_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
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
    
    print(f"Alpha-focused Excel file created: {output_path}")

def create_alpha_summary_report(folders_data, tau_count, output_path='fbd_alpha_summary_report.xlsx'):
    """
    Create a summary report focusing on alpha analysis.
    """
    df = pd.DataFrame(folders_data)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary statistics
        summary_data = {
            'Metric': [
                'Total folders scanned',
                'Folders with tau (filtered out)',
                'Folders without tau (processed)',
                'Folders with alpha values',
                'Folders without alpha values',
                'Unique alpha values found'
            ],
            'Count': [
                len(folders_data) + tau_count,
                tau_count,
                len(folders_data),
                df['alpha'].notna().sum() if 'alpha' in df.columns else 0,
                df['alpha'].isna().sum() if 'alpha' in df.columns else len(folders_data),
                df['alpha'].nunique() if 'alpha' in df.columns else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Alpha distribution
        if 'alpha' in df.columns and df['alpha'].notna().any():
            alpha_dist = df['alpha'].value_counts().sort_index()
            alpha_dist_df = pd.DataFrame({
                'Alpha Value': alpha_dist.index,
                'Experiment Count': alpha_dist.values
            })
            alpha_dist_df.to_excel(writer, sheet_name='Alpha_Distribution', index=False)
            
            # Dataset-Model matrix for each alpha
            if all(col in df.columns for col in ['dataset', 'model']):
                for alpha_value in sorted(df['alpha'].dropna().unique()):
                    alpha_df = df[df['alpha'] == alpha_value]
                    if len(alpha_df) > 0:
                        pivot = pd.crosstab(alpha_df['dataset'], alpha_df['model'], 
                                          margins=True, margins_name='Total')
                        sheet_name = f'Matrix_Alpha_{str(alpha_value).replace(".", "_")}'[:31]
                        pivot.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Alpha summary report created: {output_path}")

def main():
    """
    Main function to aggregate results focusing on alpha attribute.
    """
    # Path to fbd_run directory - adjust this for your remote server
    fbd_run_path = './fbd_run'
    
    print(f"Scanning directory: {fbd_run_path}")
    print("Filtering out folders containing 'tau'...")
    print("Extracting 'alpha' as the main attribute...")
    
    # Scan directory and extract attributes
    folders_data, tau_count = scan_and_filter_fbd_run(fbd_run_path)
    
    if not folders_data:
        print("No non-tau folders found in the specified directory!")
        return
    
    print(f"\nProcessing complete:")
    print(f"- Total tau folders filtered out: {tau_count}")
    print(f"- Non-tau folders processed: {len(folders_data)}")
    
    # Count alpha folders
    df_temp = pd.DataFrame(folders_data)
    if 'alpha' in df_temp.columns:
        alpha_count = df_temp['alpha'].notna().sum()
        print(f"- Folders with alpha values: {alpha_count}")
        if alpha_count > 0:
            print(f"- Unique alpha values: {sorted(df_temp['alpha'].dropna().unique())}")
    
    # Create Excel files
    create_alpha_focused_excel(folders_data)
    create_alpha_summary_report(folders_data, tau_count)
    
    print("\nFiles created:")
    print("- fbd_alpha_focused_results.xlsx")
    print("- fbd_alpha_summary_report.xlsx")

if __name__ == "__main__":
    main()