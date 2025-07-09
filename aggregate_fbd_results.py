import os
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict

def parse_folder_name(folder_name):
    """
    Parse folder name divided by underscores to extract attributes.
    Example: fbd_sim_bloodmnist_resnet18_20250702_202759
    """
    parts = folder_name.split('_')
    attributes = {}
    
    # Common pattern seems to be: fbd_<type>_<dataset>_<model>_<date>_<time>
    # But we'll make it flexible to handle various patterns
    
    if len(parts) >= 2:
        attributes['prefix'] = parts[0]
        attributes['type'] = parts[1]
    
    # Extract remaining parts dynamically
    for i, part in enumerate(parts[2:], start=2):
        attributes[f'attribute_{i-1}'] = part
    
    # Try to identify common patterns
    if len(parts) >= 4:
        # Check if parts look like dataset names (common medical datasets)
        medical_datasets = ['bloodmnist', 'organamnist', 'pathmnist', 'dermamnist', 
                          'octmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist',
                          'organsmnist', 'organcmnist', 'breastmnist', 'chestmnist', 'siim']
        
        # Check if parts look like model names
        model_names = ['resnet18', 'resnet50', 'vgg16', 'densenet121', 'unet', 'mobilenet']
        
        # Try to identify dataset
        for i, part in enumerate(parts):
            if any(dataset in part.lower() for dataset in medical_datasets):
                attributes['dataset'] = part
                break
        
        # Try to identify model
        for i, part in enumerate(parts):
            if any(model in part.lower() for model in model_names):
                attributes['model'] = part
                break
        
        # Check for date pattern (YYYYMMDD)
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                attributes['date'] = part
                # Check if next part is time (HHMMSS)
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                    attributes['time'] = parts[i + 1]
    
    # Add the full folder name for reference
    attributes['full_folder_name'] = folder_name
    
    return attributes

def scan_fbd_run_directory(fbd_run_path):
    """
    Scan the fbd_run directory and extract attributes from all folder names.
    """
    all_folders_data = []
    
    # Get all directories in fbd_run
    if not os.path.exists(fbd_run_path):
        print(f"Error: Directory {fbd_run_path} does not exist!")
        return all_folders_data
    
    for item in os.listdir(fbd_run_path):
        item_path = os.path.join(fbd_run_path, item)
        if os.path.isdir(item_path):
            # Parse the folder name
            attributes = parse_folder_name(item)
            
            # Try to find and read any result files in the folder
            attributes['has_results'] = check_for_results(item_path)
            
            all_folders_data.append(attributes)
    
    return all_folders_data

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

def find_unique_attributes(folders_data):
    """
    Find all unique attribute keys across all folders.
    """
    all_keys = set()
    for folder in folders_data:
        all_keys.update(folder.keys())
    return sorted(list(all_keys))

def create_summary_excel(folders_data, output_path='fbd_results_summary.xlsx'):
    """
    Create an Excel file with all folder attributes.
    """
    if not folders_data:
        print("No folders found to process!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(folders_data)
    
    # Reorder columns to put important ones first
    priority_cols = ['full_folder_name', 'type', 'dataset', 'model', 'date', 'time', 'has_results']
    other_cols = [col for col in df.columns if col not in priority_cols]
    
    # Reorder columns
    ordered_cols = [col for col in priority_cols if col in df.columns] + other_cols
    df = df[ordered_cols]
    
    # Write to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Folders', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['All_Folders']
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
    
    print(f"Summary Excel file created: {output_path}")
    
    # Also create separate sheets for each unique 'type' if exists
    if 'type' in df.columns:
        create_type_specific_excel(df, 'fbd_results_by_type.xlsx')

def create_type_specific_excel(df, output_path='fbd_results_by_type.xlsx'):
    """
    Create an Excel file with separate sheets for each experiment type.
    """
    unique_types = df['type'].unique()
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write summary sheet
        df.to_excel(writer, sheet_name='All_Types_Summary', index=False)
        
        # Write separate sheets for each type
        for exp_type in unique_types:
            if pd.notna(exp_type):  # Skip NaN values
                type_df = df[df['type'] == exp_type]
                sheet_name = str(exp_type)[:31]  # Excel sheet names limited to 31 chars
                type_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
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
    
    print(f"Type-specific Excel file created: {output_path}")

def create_dataset_model_matrix(df, output_path='fbd_dataset_model_matrix.xlsx'):
    """
    Create a matrix showing dataset vs model combinations.
    """
    if 'dataset' in df.columns and 'model' in df.columns:
        # Create a pivot table
        pivot_df = pd.crosstab(df['dataset'], df['model'], margins=True, margins_name='Total')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Dataset_Model_Matrix')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Dataset_Model_Matrix']
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
        
        print(f"Dataset-Model matrix Excel file created: {output_path}")

def main():
    """
    Main function to aggregate results from fbd_run directory.
    """
    # You can modify this path to match your remote server structure
    fbd_run_path = './fbd_run'
    
    print(f"Scanning directory: {fbd_run_path}")
    
    # Scan directory and extract attributes
    folders_data = scan_fbd_run_directory(fbd_run_path)
    
    if not folders_data:
        print("No folders found in the specified directory!")
        return
    
    print(f"Found {len(folders_data)} folders")
    
    # Find unique attributes
    unique_attrs = find_unique_attributes(folders_data)
    print(f"Unique attributes found: {unique_attrs}")
    
    # Create summary Excel file
    create_summary_excel(folders_data)
    
    # Create dataset-model matrix if applicable
    df = pd.DataFrame(folders_data)
    create_dataset_model_matrix(df)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if 'type' in df.columns:
        print("\nExperiment types:")
        print(df['type'].value_counts())
    
    if 'dataset' in df.columns:
        print("\nDatasets:")
        print(df['dataset'].value_counts())
    
    if 'model' in df.columns:
        print("\nModels:")
        print(df['model'].value_counts())
    
    print(f"\nTotal folders processed: {len(folders_data)}")

if __name__ == "__main__":
    main()