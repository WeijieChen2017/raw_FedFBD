import pandas as pd

# Read the Excel file
df = pd.read_excel('fbd_results_summary.xlsx')

# Filter out tau folders
df_non_tau = df[~df['full_folder_name'].str.lower().str.contains('tau', na=False)]

print("Sample folder names (non-tau):")
print("="*80)
for i, name in enumerate(df_non_tau['full_folder_name'].head(20)):
    print(f"{i+1}: {name}")

print(f"\nTotal non-tau folders: {len(df_non_tau)}")
print(f"Total folders with tau: {len(df) - len(df_non_tau)}")

# Look for patterns that might contain alpha
print("\n\nLooking for potential alpha patterns:")
print("="*80)
for name in df_non_tau['full_folder_name']:
    if any(keyword in name.lower() for keyword in ['alpha', 'alph', 'a_', '_a']):
        print(name)