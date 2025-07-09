import os
import pandas as pd
from pathlib import Path
import json
import re
from collections import defaultdict

def extract_alpha_from_folder_name(folder_name):
    """
    Extract alpha value from folder name.
    Example patterns:
    - ...alpha_0.25_...
    - ...alpha0.25...
    - ...alpha=0.25...
    """
    # Try different patterns to extract alpha value
    patterns = [
        r'alpha[_\-=]?(\d+\.?\d*)',  # alpha_0.25, alpha0.25, alpha=0.25
        r'alpha[_\-=]?(\d+)',         # alpha_25 (might mean 0.25)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            alpha_str = match.group(1)
            try:
                # Convert to float
                alpha = float(alpha_str)
                # If it's a large number like 25, assume it means 0.25
                if alpha > 1:
                    alpha = alpha / 100
                return alpha
            except:
                pass
    
    return None

def parse_folder_with_alpha(folder_name):
    """
    Parse folder name to extract attributes with focus on alpha.
    """
    attributes = {}
    
    # Extract alpha value
    alpha = extract_alpha_from_folder_name(folder_name)
    if alpha is not None:
        attributes['alpha'] = alpha
    
    # Parse by underscores
    parts = folder_name.split('_')
    
    # Common patterns
    attributes['full_folder_name'] = folder_name
    
    # Extract dataset
    medical_datasets = ['bloodmnist', 'organamnist', 'pathmnist', 'dermamnist', 
                      'octmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist',
                      'organsmnist', 'organcmnist', 'breastmnist', 'chestmnist', 'siim']
    
    for part in parts:
        if any(dataset in part.lower() for dataset in medical_datasets):
            attributes['dataset'] = part
            break
    
    # Extract model
    model_names = ['resnet18', 'resnet50', 'vgg16', 'densenet121', 'unet', 'mobilenet']
    
    for part in parts:
        if any(model in part.lower() for model in model_names):
            attributes['model'] = part
            break
    
    # Extract regularization type
    if 'reg' in parts:
        reg_index = parts.index('reg')
        if reg_index + 1 < len(parts):
            attributes['reg_type'] = parts[reg_index + 1]
    
    # Extract coefficient
    for i, part in enumerate(parts):
        if 'coef' in part and i + 1 < len(parts):
            try:
                attributes['coef'] = float(parts[i + 1])
            except:
                pass
    
    # Extract fold number for SIIM
    fold_match = re.search(r'fold[_\-]?(\d+)', folder_name)
    if fold_match:
        attributes['fold'] = int(fold_match.group(1))
    
    # Extract date and time
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            attributes['date'] = part
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                attributes['time'] = parts[i + 1]
    
    return attributes

def filter_non_tau_folders(folders_data):
    """
    Filter out folders containing 'tau' in their names.
    """
    filtered_data = []
    
    for folder in folders_data:
        folder_name = folder.get('full_folder_name', '')
        if 'tau' not in folder_name.lower():
            filtered_data.append(folder)
    
    return filtered_data

def process_existing_excel(input_file='fbd_results_summary.xlsx'):
    """
    Process existing Excel file to filter out tau folders and extract alpha values.
    """
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Filter out rows with 'tau' in folder name
    df_filtered = df[~df['full_folder_name'].str.lower().str.contains('tau', na=False)]
    
    # Extract alpha values
    df_filtered['alpha'] = df_filtered['full_folder_name'].apply(extract_alpha_from_folder_name)
    
    # Re-parse folder names for better attribute extraction
    parsed_data = []
    for _, row in df_filtered.iterrows():
        folder_name = row['full_folder_name']
        attributes = parse_folder_with_alpha(folder_name)
        
        # Merge with existing data
        for col in df_filtered.columns:
            if col not in attributes and pd.notna(row[col]):
                attributes[col] = row[col]
        
        parsed_data.append(attributes)
    
    return pd.DataFrame(parsed_data)

def create_alpha_grouped_excel(df, output_path='fbd_alpha_analysis.xlsx'):
    """
    Create Excel file with sheets grouped by alpha values.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write summary sheet
        df_summary = df.copy()
        
        # Reorder columns to put alpha first
        if 'alpha' in df_summary.columns:
            priority_cols = ['alpha', 'full_folder_name', 'dataset', 'model', 'reg_type', 'coef', 'fold', 'date', 'time']
            other_cols = [col for col in df_summary.columns if col not in priority_cols]
            ordered_cols = [col for col in priority_cols if col in df_summary.columns] + other_cols
            df_summary = df_summary[ordered_cols]
        
        df_summary.to_excel(writer, sheet_name='All_Non_Tau_Results', index=False)
        
        # Create sheets for each alpha value
        if 'alpha' in df.columns:
            # Group by alpha
            alpha_groups = df.groupby('alpha')
            
            for alpha_value, group_df in alpha_groups:
                if pd.notna(alpha_value):
                    sheet_name = f'alpha_{alpha_value}'.replace('.', '_')[:31]
                    group_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
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
        
        # Auto-adjust summary sheet columns
        worksheet = writer.sheets['All_Non_Tau_Results']
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
    
    print(f"Alpha-grouped Excel file created: {output_path}")

def create_alpha_summary_stats(df, output_path='fbd_alpha_summary_stats.xlsx'):
    """
    Create summary statistics grouped by alpha values.
    """
    if 'alpha' not in df.columns:
        print("No alpha values found in the data!")
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Alpha distribution
        alpha_stats = df.groupby('alpha').agg({
            'full_folder_name': 'count',
            'dataset': lambda x: x.nunique() if 'dataset' in df.columns else 0,
            'model': lambda x: x.nunique() if 'model' in df.columns else 0,
        }).rename(columns={
            'full_folder_name': 'experiment_count',
            'dataset': 'unique_datasets',
            'model': 'unique_models'
        })
        
        alpha_stats.to_excel(writer, sheet_name='Alpha_Statistics')
        
        # Dataset-Model matrix for each alpha
        if 'dataset' in df.columns and 'model' in df.columns:
            for alpha_value in sorted(df['alpha'].dropna().unique()):
                alpha_df = df[df['alpha'] == alpha_value]
                if len(alpha_df) > 0:
                    pivot = pd.crosstab(alpha_df['dataset'], alpha_df['model'], margins=True, margins_name='Total')
                    sheet_name = f'Matrix_alpha_{alpha_value}'.replace('.', '_')[:31]
                    pivot.to_excel(writer, sheet_name=sheet_name)
        
        # Overall summary
        summary_data = {
            'Total Experiments (non-tau)': [len(df)],
            'Experiments with Alpha': [df['alpha'].notna().sum()],
            'Unique Alpha Values': [df['alpha'].nunique()],
            'Alpha Values': [', '.join(map(str, sorted(df['alpha'].dropna().unique())))]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Alpha summary statistics created: {output_path}")

def main():
    """
    Main function to process alpha-focused results.
    """
    print("Processing existing Excel files to extract alpha-focused results...")
    
    # Check if the summary file exists
    if os.path.exists('fbd_results_summary.xlsx'):
        # Process existing Excel file
        df = process_existing_excel('fbd_results_summary.xlsx')
        
        print(f"\nTotal folders (after filtering tau): {len(df)}")
        
        if 'alpha' in df.columns:
            print(f"Folders with alpha values: {df['alpha'].notna().sum()}")
            print(f"Unique alpha values: {sorted(df['alpha'].dropna().unique())}")
        
        # Create alpha-grouped Excel
        create_alpha_grouped_excel(df)
        
        # Create summary statistics
        create_alpha_summary_stats(df)
        
        # Print summary
        print("\n=== Alpha Distribution ===")
        if 'alpha' in df.columns:
            alpha_counts = df['alpha'].value_counts().sort_index()
            for alpha, count in alpha_counts.items():
                print(f"Alpha {alpha}: {count} experiments")
    else:
        print("Error: fbd_results_summary.xlsx not found!")
        print("Please run aggregate_fbd_results.py first to generate the summary file.")

if __name__ == "__main__":
    main()