#!/usr/bin/env python3
import json
import re
import os
import tempfile
import shutil

def update_labels_in_file(filename):
    """Update all label paths in a JSON file to add '_mor' before '_128.nii.gz'"""
    print(f"Processing {filename}...")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {filename}: {e}")
        return
    
    updated_count = 0

    def update_paths(obj):
        nonlocal updated_count
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'label' and isinstance(value, str) and "_mor_mor_mor_128.nii.gz" in value:
                    new_value = value.replace("_mor_mor_mor_128.nii.gz", "_mor_128.nii.gz")
                    if new_value != value:
                        obj[key] = new_value
                        updated_count += 1
                else:
                    update_paths(value)
        elif isinstance(obj, list):
            for item in obj:
                update_paths(item)
    
    update_paths(data)
    
    # Use a temporary file to ensure atomic write.
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename))
    
    try:
        with os.fdopen(temp_fd, 'w') as temp_f:
            json.dump(data, temp_f, indent=2)
        
        shutil.move(temp_path, filename)
        print(f"Updated {updated_count} labels in {filename}")
    except Exception as e:
        print(f"Error writing changes to {filename}: {e}")
    finally:
        # Ensure the temporary file is removed if the move fails
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    # Get the directory where the script is located to make it runnable from anywhere.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Update all SIIM fold files
    fold_files = [
        'siim_fbd_fold_0.json',
        'siim_fbd_fold_1.json', 
        'siim_fbd_fold_2.json',
        'siim_fbd_fold_3.json'
    ]
    
    for filename in fold_files:
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            update_labels_in_file(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    print("All files processed!")

if __name__ == "__main__":
    main() 