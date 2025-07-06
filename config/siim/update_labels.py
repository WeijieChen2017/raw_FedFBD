#!/usr/bin/env python3
import json
import re
import os

def update_labels_in_file(filename):
    """Update all label paths in a JSON file to add '_mor' before '_128.nii.gz'"""
    print(f"Processing {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Pattern to match labels/XX_128.nii.gz and replace with labels/XX_mor_128.nii.gz
    pattern = r'labels/(\d+)_128\.nii\.gz'
    replacement = r'labels/\1_mor_128.nii.gz'
    
    def update_paths(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'label' and isinstance(value, str):
                    obj[key] = re.sub(pattern, replacement, value)
                else:
                    update_paths(value)
        elif isinstance(obj, list):
            for item in obj:
                update_paths(item)
    
    update_paths(data)
    
    # Write back to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {filename}")

def main():
    # Update all SIIM fold files
    fold_files = [
        'siim_fbd_fold_0.json',
        'siim_fbd_fold_1.json', 
        'siim_fbd_fold_2.json',
        'siim_fbd_fold_3.json'
    ]
    
    for filename in fold_files:
        if os.path.exists(filename):
            update_labels_in_file(filename)
        else:
            print(f"Warning: {filename} not found")
    
    print("All files processed!")

if __name__ == "__main__":
    main() 