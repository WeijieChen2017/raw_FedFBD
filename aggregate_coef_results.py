import os
import pandas as pd
from pathlib import Path
import json
import re
from collections import defaultdict

def extract_coef_from_folder_name(folder_name):
    """
    Extract coefficient value from folder name.
    Example patterns:
    - ...coef_0.100_...
    - ...coef_1.000_...
    """
    match = re.search(r'coef[_\-](\d+\.?\d*)', folder_name)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    return None

def extract_al_type(folder_name):
    """
    Extract AL (Active Learning) type from folder name.
    Example patterns:
    - ..._al_ent (entropy-based)
    - ..._al_al (another AL method)
    - ..._al (generic AL)
    """
    if '_al_ent' in folder_name:
        return 'entropy'
    elif '_al_al' in folder_name:
        return 'al_method'
    elif folder_name.endswith('_al'):
        return 'generic'
    return None

def parse_folder_comprehensive(folder_name):
    """
    Parse folder name to extract all relevant attributes.
    """
    attributes = {}
    attributes['full_folder_name'] = folder_name
    
    # Extract coefficient (which might be the "alpha" you're looking for)
    coef = extract_coef_from_folder_name(folder_name)
    if coef is not None:
        attributes['coef_alpha'] = coef
    
    # Extract AL type
    al_type = extract_al_type(folder_name)
    if al_type:
        attributes['al_type'] = al_type
    
    # Parse by underscores
    parts = folder_name.split('_')
    
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

def process_excel_with_coef_focus(input_file='fbd_results_summary.xlsx'):
    """
    Process existing Excel file to filter out tau folders and focus on coefficient values.
    """
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Filter out rows with 'tau' in folder name
    df_filtered = df[~df['full_folder_name'].str.lower().str.contains('tau', na=False)].copy()
    
    # Re-parse folder names for better attribute extraction
    parsed_data = []
    for _, row in df_filtered.iterrows():
        folder_name = row['full_folder_name']
        attributes = parse_folder_comprehensive(folder_name)
        
        # Merge with existing data
        for col in df_filtered.columns:
            if col not in attributes and pd.notna(row[col]):
                attributes[col] = row[col]
        
        parsed_data.append(attributes)
    
    return pd.DataFrame(parsed_data)

def create_coef_grouped_excel(df, output_path='fbd_coefficient_analysis.xlsx'):
    """
    Create Excel file with sheets grouped by coefficient (alpha) values.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write summary sheet
        df_summary = df.copy()
        
        # Reorder columns
        priority_cols = ['coef_alpha', 'full_folder_name', 'dataset', 'model', 'reg_type', 
                        'al_type', 'fold', 'date', 'time']
        other_cols = [col for col in df_summary.columns if col not in priority_cols]
        ordered_cols = [col for col in priority_cols if col in df_summary.columns] + other_cols
        df_summary = df_summary[ordered_cols]
        
        df_summary.to_excel(writer, sheet_name='All_Non_Tau_Results', index=False)
        
        # Create sheets for each coefficient value
        if 'coef_alpha' in df.columns:
            # Group by coefficient
            coef_groups = df.groupby('coef_alpha')
            
            for coef_value, group_df in coef_groups:
                if pd.notna(coef_value):
                    sheet_name = f'coef_{coef_value}'.replace('.', '_')[:31]
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
        
        # Create sheet for AL methods if present
        if 'al_type' in df.columns and df['al_type'].notna().any():
            al_df = df[df['al_type'].notna()]
            al_df.to_excel(writer, sheet_name='Active_Learning_Experiments', index=False)
    
    print(f"Coefficient-grouped Excel file created: {output_path}")

def create_coef_summary_stats(df, output_path='fbd_coefficient_summary_stats.xlsx'):
    """
    Create summary statistics grouped by coefficient values.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Overall summary
        summary_data = {
            'Metric': ['Total Experiments (non-tau)', 'Experiments with Coefficient', 
                      'Experiments with AL', 'Unique Coefficient Values', 'Unique Datasets', 'Unique Models'],
            'Value': [
                len(df),
                df['coef_alpha'].notna().sum() if 'coef_alpha' in df.columns else 0,
                df['al_type'].notna().sum() if 'al_type' in df.columns else 0,
                df['coef_alpha'].nunique() if 'coef_alpha' in df.columns else 0,
                df['dataset'].nunique() if 'dataset' in df.columns else 0,
                df['model'].nunique() if 'model' in df.columns else 0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Coefficient distribution
        if 'coef_alpha' in df.columns:
            coef_stats = df.groupby('coef_alpha').agg({
                'full_folder_name': 'count',
                'dataset': lambda x: x.nunique() if 'dataset' in df.columns else 0,
                'model': lambda x: x.nunique() if 'model' in df.columns else 0,
                'al_type': lambda x: x.notna().sum() if 'al_type' in df.columns else 0,
            }).rename(columns={
                'full_folder_name': 'experiment_count',
                'dataset': 'unique_datasets',
                'model': 'unique_models',
                'al_type': 'al_experiments'
            })
            
            coef_stats.to_excel(writer, sheet_name='Coefficient_Statistics')
        
        # Dataset-Model matrix for each coefficient
        if all(col in df.columns for col in ['dataset', 'model', 'coef_alpha']):
            for coef_value in sorted(df['coef_alpha'].dropna().unique()):
                coef_df = df[df['coef_alpha'] == coef_value]
                if len(coef_df) > 0:
                    pivot = pd.crosstab(coef_df['dataset'], coef_df['model'], margins=True, margins_name='Total')
                    sheet_name = f'Matrix_coef_{coef_value}'.replace('.', '_')[:31]
                    pivot.to_excel(writer, sheet_name=sheet_name)
        
        # AL method distribution
        if 'al_type' in df.columns and df['al_type'].notna().any():
            al_stats = df[df['al_type'].notna()].groupby(['al_type', 'coef_alpha']).size().unstack(fill_value=0)
            al_stats.to_excel(writer, sheet_name='AL_Distribution')
    
    print(f"Coefficient summary statistics created: {output_path}")

def main():
    """
    Main function to process coefficient-focused results.
    """
    print("Processing existing Excel files to extract coefficient-focused results...")
    
    # Check if the summary file exists
    if os.path.exists('fbd_results_summary.xlsx'):
        # Process existing Excel file
        df = process_excel_with_coef_focus('fbd_results_summary.xlsx')
        
        print(f"\nTotal folders (after filtering tau): {len(df)}")
        
        if 'coef_alpha' in df.columns:
            print(f"Folders with coefficient values: {df['coef_alpha'].notna().sum()}")
            print(f"Unique coefficient values: {sorted(df['coef_alpha'].dropna().unique())}")
        
        if 'al_type' in df.columns:
            print(f"Folders with AL methods: {df['al_type'].notna().sum()}")
            print(f"AL types: {df['al_type'].value_counts().to_dict()}")
        
        # Create coefficient-grouped Excel
        create_coef_grouped_excel(df)
        
        # Create summary statistics
        create_coef_summary_stats(df)
        
        # Print summary
        print("\n=== Coefficient (Alpha) Distribution ===")
        if 'coef_alpha' in df.columns:
            coef_counts = df['coef_alpha'].value_counts().sort_index()
            for coef, count in coef_counts.items():
                print(f"Coefficient {coef}: {count} experiments")
    else:
        print("Error: fbd_results_summary.xlsx not found!")
        print("Please run aggregate_fbd_results.py first to generate the summary file.")

if __name__ == "__main__":
    main()