import medmnist
from medmnist import INFO
import os

# Create output directory for montages
output_dir = "montages"
os.makedirs(output_dir, exist_ok=True)

# Get all 2D datasets (exclude 3D datasets)
all_2d_datasets = []
for flag, info in INFO.items():
    if '3d' not in flag.lower():
        all_2d_datasets.append(flag)

print(f"Found {len(all_2d_datasets)} 2D datasets: {all_2d_datasets}")

# Create montages for each dataset
for data_flag in all_2d_datasets:
    print(f"\nProcessing {data_flag}...")
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load dataset from cache
    try:
        test_dataset = DataClass(split='test', download=False, root='./cache/')
    except Exception as e:
        print(f"Skipping {data_flag}: {str(e)}")
        continue
    
    # Generate montage using native method (length=4 means 4x4 grid = 16 images)
    try:
        test_dataset.montage(length=4, save_folder=output_dir)
        print(f"Generated montage for {data_flag}")
    except Exception as e:
        print(f"Error generating montage for {data_flag}: {str(e)}")

print(f"\nAll montages saved to '{output_dir}/' directory") 