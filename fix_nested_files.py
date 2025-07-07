#!/usr/bin/env python3
"""
Script to fix nested SIIM dataset file structure.
Moves files from nested directories like data/21_128.nii.gz/21_128.nii.gz 
to the expected location data/21_128.nii.gz
"""

import os
import shutil
import glob

def fix_nested_files(data_root="siim-101"):
    """
    Fix nested file structure in SIIM dataset.
    
    Args:
        data_root: Root directory containing SIIM data
    """
    print(f"Fixing nested files in {data_root}...")
    
    # Find all .nii.gz directories (these should be files, not directories)
    pattern = os.path.join(data_root, "**", "*.nii.gz")
    nested_dirs = []
    
    for path in glob.glob(pattern, recursive=True):
        if os.path.isdir(path):
            nested_dirs.append(path)
    
    print(f"Found {len(nested_dirs)} nested directories to fix")
    
    fixed_count = 0
    errors = []
    
    for nested_dir in nested_dirs:
        try:
            # Expected filename from the directory name
            expected_filename = os.path.basename(nested_dir)
            
            # Check if the file exists inside the nested directory
            nested_file = os.path.join(nested_dir, expected_filename)
            
            if os.path.isfile(nested_file):
                print(f"Moving: {nested_file} -> {nested_dir}")
                
                # Create a temporary name to avoid conflicts
                temp_name = nested_dir + "_temp"
                
                # Move the file to temporary location
                shutil.move(nested_file, temp_name)
                
                # Remove the now-empty directory
                try:
                    os.rmdir(nested_dir)
                except OSError as e:
                    print(f"Warning: Could not remove directory {nested_dir}: {e}")
                    # Try to remove any remaining files
                    try:
                        shutil.rmtree(nested_dir)
                    except:
                        pass
                
                # Move the file to the correct location
                shutil.move(temp_name, nested_dir)
                
                fixed_count += 1
                print(f"  ✓ Fixed: {expected_filename}")
            else:
                print(f"  ✗ File not found in nested directory: {nested_file}")
                errors.append(f"File not found: {nested_file}")
                
        except Exception as e:
            error_msg = f"Error processing {nested_dir}: {e}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
    
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count} files")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
    
    return fixed_count, errors

def main():
    """Main function"""
    print("SIIM Dataset Nested File Structure Fix")
    print("=" * 40)
    
    # Check if data directory exists
    data_root = "siim-101"
    if not os.path.exists(data_root):
        print(f"Error: Data directory '{data_root}' not found!")
        print("Please ensure you're running this script from the correct directory.")
        return 1
    
    # Run the fix
    fixed_count, errors = fix_nested_files(data_root)
    
    if errors:
        print(f"\nCompleted with {len(errors)} errors. Please check the output above.")
        return 1
    else:
        print(f"\nSuccessfully fixed {fixed_count} files!")
        return 0

if __name__ == "__main__":
    exit(main())