#!/usr/bin/env python3

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def extract_site_and_patient_info(file_path):
    """Extract site number and patient ID from file path"""
    path_parts = Path(file_path).parts
    
    # Find site information
    site_match = None
    for part in path_parts:
        if 'Site_' in part:
            site_match = re.search(r'Site_(\d+)', part)
            break
    
    site_id = site_match.group(1) if site_match else "Unknown"
    
    # Extract patient ID from filename
    filename = Path(file_path).stem
    if filename.endswith('.nii'):
        filename = filename[:-4]
    
    # Patient ID is typically the part before file type indicators
    patient_id = filename.split('_')[0] if '_' in filename else filename
    
    return site_id, patient_id

def analyze_nifti_file(file_path):
    """Analyze a single NIfTI file and return its properties"""
    try:
        # Load the NIfTI file
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        
        # Get basic properties
        shape = data.shape
        voxel_sizes = header.get_zooms()
        data_type = data.dtype
        
        # Get intensity statistics
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Check if it's likely a label/mask (binary or small integer values)
        unique_vals = np.unique(data)
        is_binary = len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))
        is_label = len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int))
        
        # Get file info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        site_id, patient_id = extract_site_and_patient_info(file_path)
        
        # Determine file type based on path
        file_type = "Unknown"
        if "_mask" in file_path or "_label" in file_path or "_seg" in file_path:
            file_type = "Label/Mask"
        elif "_image" in file_path or "_img" in file_path:
            file_type = "Image"
        elif is_label:
            file_type = "Label/Mask"
        else:
            file_type = "Image"
        
        return {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'site_id': site_id,
            'patient_id': patient_id,
            'file_type': file_type,
            'shape_x': shape[0],
            'shape_y': shape[1],
            'shape_z': shape[2] if len(shape) > 2 else 1,
            'voxel_size_x': voxel_sizes[0],
            'voxel_size_y': voxel_sizes[1],
            'voxel_size_z': voxel_sizes[2] if len(voxel_sizes) > 2 else 1.0,
            'data_type': str(data_type),
            'file_size_mb': file_size_mb,
            'min_value': min_val,
            'max_value': max_val,
            'mean_value': mean_val,
            'std_value': std_val,
            'unique_values_count': len(unique_vals),
            'is_binary': is_binary,
            'is_likely_label': is_label,
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'site_id': "Error",
            'patient_id': "Error",
            'file_type': "Error",
            'error': str(e)
        }

def find_nifti_files(base_dir):
    """Find all .nii.gz files in the dataset"""
    pattern = os.path.join(base_dir, "**", "*.nii.gz")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

def create_summary_statistics(df):
    """Create summary statistics from the analyzed data"""
    summaries = []
    
    # Overall summary
    total_files = len(df)
    sites = df['site_id'].unique()
    
    # Summary by site
    for site in sites:
        site_data = df[df['site_id'] == site]
        
        # Summary by file type within site
        for file_type in site_data['file_type'].unique():
            type_data = site_data[site_data['file_type'] == file_type]
            
            if len(type_data) > 0 and 'error' not in type_data.columns:
                summary = {
                    'site_id': site,
                    'file_type': file_type,
                    'count': len(type_data),
                    'unique_patients': type_data['patient_id'].nunique(),
                    'avg_shape_x': type_data['shape_x'].mean(),
                    'avg_shape_y': type_data['shape_y'].mean(),
                    'avg_shape_z': type_data['shape_z'].mean(),
                    'min_shape_x': type_data['shape_x'].min(),
                    'max_shape_x': type_data['shape_x'].max(),
                    'min_shape_y': type_data['shape_y'].min(),
                    'max_shape_y': type_data['shape_y'].max(),
                    'min_shape_z': type_data['shape_z'].min(),
                    'max_shape_z': type_data['shape_z'].max(),
                    'avg_file_size_mb': type_data['file_size_mb'].mean(),
                    'total_size_mb': type_data['file_size_mb'].sum(),
                }
                summaries.append(summary)
    
    return pd.DataFrame(summaries)

def main():
    parser = argparse.ArgumentParser(description="Analyze SIIM dataset NIfTI files and create Excel summary")
    parser.add_argument("--data_dir", type=str, 
                        default="siim-101/SIIM_Fed_Learning_Phase1Data",
                        help="Path to SIIM dataset directory")
    parser.add_argument("--output_file", type=str, 
                        default="siim_dataset_analysis.xlsx",
                        help="Output Excel file name")
    parser.add_argument("--include_errors", action="store_true",
                        help="Include files that couldn't be analyzed in the output")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("Please check the path or run from the correct directory.")
        return 1
    
    print(f"Analyzing SIIM dataset in: {args.data_dir}")
    
    # Find all NIfTI files
    print("Finding all .nii.gz files...")
    nifti_files = find_nifti_files(args.data_dir)
    
    if not nifti_files:
        print(f"No .nii.gz files found in {args.data_dir}")
        return 1
    
    print(f"Found {len(nifti_files)} .nii.gz files")
    
    # Analyze each file
    print("Analyzing files...")
    results = []
    errors = []
    
    for file_path in tqdm(nifti_files, desc="Processing files"):
        result = analyze_nifti_file(file_path)
        if 'error' in result:
            errors.append(result)
        else:
            results.append(result)
    
    # Create DataFrames
    df_main = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors) if errors else pd.DataFrame()
    
    print(f"Successfully analyzed: {len(results)} files")
    if errors:
        print(f"Errors encountered: {len(errors)} files")
    
    # Create summary statistics
    if len(df_main) > 0:
        print("Creating summary statistics...")
        df_summary = create_summary_statistics(df_main)
        
        # Create Excel file with multiple sheets
        print(f"Saving results to {args.output_file}...")
        with pd.ExcelWriter(args.output_file, engine='openpyxl') as writer:
            # Main detailed data
            df_main.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Summary statistics
            df_summary.to_excel(writer, sheet_name='Summary_by_Site_Type', index=False)
            
            # Shape distribution
            shape_counts = df_main.groupby(['shape_x', 'shape_y', 'shape_z']).size().reset_index(name='count')
            shape_counts = shape_counts.sort_values('count', ascending=False)
            shape_counts.to_excel(writer, sheet_name='Shape_Distribution', index=False)
            
            # Site overview
            site_overview = df_main.groupby('site_id').agg({
                'patient_id': 'nunique',
                'file_type': lambda x: ', '.join(x.unique()),
                'file_size_mb': ['count', 'sum', 'mean']
            }).round(2)
            site_overview.columns = ['Unique_Patients', 'File_Types', 'Total_Files', 'Total_Size_MB', 'Avg_File_Size_MB']
            site_overview.to_excel(writer, sheet_name='Site_Overview')
            
            # Errors sheet (if any)
            if len(df_errors) > 0 and args.include_errors:
                df_errors.to_excel(writer, sheet_name='Errors', index=False)
        
        print(f"Excel file saved successfully: {args.output_file}")
        
        # Print quick summary to console
        print("\n=== Quick Summary ===")
        print(f"Total files analyzed: {len(df_main)}")
        print(f"Total sites: {df_main['site_id'].nunique()}")
        print(f"Total patients: {df_main['patient_id'].nunique()}")
        print(f"File types found: {', '.join(df_main['file_type'].unique())}")
        print(f"Shape variations: {len(shape_counts)}")
        print(f"Most common shape: {shape_counts.iloc[0]['shape_x']}x{shape_counts.iloc[0]['shape_y']}x{shape_counts.iloc[0]['shape_z']} ({shape_counts.iloc[0]['count']} files)")
        
        if errors:
            print(f"Files with errors: {len(errors)}")
    
    else:
        print("No files could be analyzed successfully.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())