import pandas as pd

# Check the structure of the Excel files
files = [
    'fbd_results_summary.xlsx',
    'fbd_results_by_type.xlsx',
    'fbd_dataset_model_matrix.xlsx'
]

for file in files:
    try:
        print(f"\n{'='*60}")
        print(f"File: {file}")
        print(f"{'='*60}")
        
        # Read the first sheet
        df = pd.read_excel(file)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst 5 rows of 'full_folder_name' column (if exists):")
        if 'full_folder_name' in df.columns:
            print(df['full_folder_name'].head(10))
        
        # Check for multiple sheets
        xl_file = pd.ExcelFile(file)
        if len(xl_file.sheet_names) > 1:
            print(f"\nSheet names: {xl_file.sheet_names}")
            
    except Exception as e:
        print(f"Error reading {file}: {e}")