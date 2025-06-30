import os
import argparse
import subprocess
import shutil
import hashlib

DATASETS_2D = [
    "bloodmnist", # no specific rules
    "breastmnist", # --as_rgb
    "chestmnist", # --as_rgb
    "dermamnist", # no specific rules
    "octmnist", # --as_rgb
    "organamnist", # --as_rgb
    "organcmnist", # --as_rgb
    "organsmnist", # --as_rgb
    "pathmnist", # no specific rules
    "pneumoniamnist", # --as_rgb
    "retinamnist", # no specific rules
    "tissuemnist", # --as_rgb
]

DATASETS_3D = [
    "adrenalmnist3d", # --as_rgb
    "fracturemnist3d", # --as_rgb
    "nodulemnist3d", # --as_rgb
    "organmnist3d", # --as_rgb
    "synapsemnist3d", # --as_rgb
    "vesselmnist3d", # --as_rgb
]

DATASET_SPECIFIC_RULES = {
    "breastmnist": "--as_rgb",
    "adrenalmnist3d": "--as_rgb",
    "octmnist": "--as_rgb",
    "organcmnist": "--as_rgb",
    "tissuemnist": "--as_rgb",
    "pneumoniamnist": "--as_rgb",
    "chestmnist": "--as_rgb",
    "organamnist": "--as_rgb",
    "organsmnist": "--as_rgb",
    "fracturemnist3d": "--as_rgb",
    "nodulemnist3d": "--as_rgb",
    "organmnist3d": "--as_rgb",
    "synapsemnist3d": "--as_rgb",
    "vesselmnist3d": "--as_rgb",
}

MODEL_INDEX = [1, 2, 3]
NUM_PARTS = 6
CACHE_DIR = "data_storage"
MEDMNIST_DIR = os.path.expanduser("/root/.medmnist")
BATCH_SIZE_2D_DEFAULT = 128 
BATCH_SIZE_3D_DEFAULT = 16
BATCH_SIZE_EXPAND_FACTOR_2D = 12
BATCH_SIZE_EXPAND_FACTOR_3D = 12

def handle_dataset_cache(dataset, post_execution=False):
    """Manages the dataset cache by copying files when needed."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(MEDMNIST_DIR):
        os.makedirs(MEDMNIST_DIR)
        
    source_npz_path = os.path.join(CACHE_DIR, f"{dataset}.npz")
    dest_npz_path = os.path.join(MEDMNIST_DIR, f"{dataset}.npz")

    if not post_execution:
        print(f"Looking for {dataset} in {CACHE_DIR}")
        # Before execution: if file is in cache but not in destination, copy it.
        if os.path.exists(source_npz_path):
            if not os.path.exists(dest_npz_path):
                print(f"Found {dataset} in cache. Copying to {MEDMNIST_DIR}")
                shutil.copy(source_npz_path, dest_npz_path)
            else:
                # please have the md5 check here
                md5_hash = hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest()
                if md5_hash == hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest():
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} and is the same")
                else:
                    print(f"Dataset {dataset} already exists in {MEDMNIST_DIR} but is different")
                    shutil.copy(source_npz_path, dest_npz_path)
        else:
            print(f"Dataset {dataset} not found in cache {CACHE_DIR}")
    else:
        # Copy downloaded dataset to cache if not already there
        if os.path.exists(dest_npz_path):
            if not os.path.exists(source_npz_path):
                print(f"Copying {dataset} from {MEDMNIST_DIR} to cache")
                shutil.copy(dest_npz_path, source_npz_path)
            else:
                md5_hash = hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest()
                if md5_hash == hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest():
                    print(f"Dataset {dataset} already exists in cache and is the same")
                else:
                    print(f"Dataset {dataset} already exists in cache but is different")
                    shutil.copy(dest_npz_path, source_npz_path)

def main():
    """Run MedMNIST evaluation for a specific fold."""
    parser = argparse.ArgumentParser(description="Run MedMNIST evaluation for a specific fold.")
    parser.add_argument("--fold", type=int, default=0, choices=range(NUM_PARTS),
                        help=f"The fold to run, from 0 to {NUM_PARTS-1}.")
    parser.add_argument("--dataset", type=str, default=None, help="Run for a specific dataset.")
    args = parser.parse_args()

    fold = args.fold
    if args.dataset:
        if args.dataset in DATASETS_2D:
            datasets_2d = [args.dataset]
            datasets_3d = []
            print(f"Running 2D evaluation for {args.dataset}")
        elif args.dataset in DATASETS_3D:
            datasets_2d = []
            datasets_3d = [args.dataset]
            print(f"Running 3D evaluation for {args.dataset}")
        else:
            print(f"Error: Dataset '{args.dataset}' not found.")
            return
    else:
        datasets_2d = DATASETS_2D[fold*2:(fold+1)*2]  # 12 datasets split into 6 parts of 2
        datasets_3d = DATASETS_3D[fold:(fold+1)]  # 6 datasets split into 6 parts of 1
        print(f"Running 2D evaluation for {datasets_2d} with fold {fold}")
        print(f"Running 3D evaluation for {datasets_3d} with fold {fold}")

    datasets = {'2d': datasets_2d, '3d': datasets_3d}

    print(f"================================================")
    # 2D commands
    IMAGE_SIZES = [28, 224]
    MODEL_NAMES = ["resnet18", "resnet50"]
    script_path = "experiments/MedMNIST2D/train_and_eval_pytorch.py"
    for dataset in datasets['2d']:
        for model_name in MODEL_NAMES:
            for image_size in IMAGE_SIZES:
                for model_index in MODEL_INDEX:
                    command = (
                        f"python {script_path} "
                        f"--data_flag {dataset} "
                        f"--model_path weights/weights_{dataset}/{model_name}_{image_size}_{model_index}.pth "
                        f"--model_flag {model_name} --size {image_size} --num_epochs 0 --download --run {model_index} --batch_size {BATCH_SIZE_2D_DEFAULT * BATCH_SIZE_EXPAND_FACTOR_2D} "
                        f"--gpu_ids 0"
                    )
                    if image_size == 224:
                        command += " --resize"
                    if dataset in DATASET_SPECIFIC_RULES:
                        command += f" {DATASET_SPECIFIC_RULES[dataset]}"
                    
                    dataset_cache_name = f"{dataset}" if image_size == 28 else f"{dataset}_{image_size}"
                    handle_dataset_cache(dataset_cache_name)
                    print(f"Executing: {command}")
                    subprocess.run(command, shell=True, check=True)
                    handle_dataset_cache(dataset_cache_name, post_execution=True)
                    print(f"--------------------------------")

    # 3D commands
    MODEL_NAMES = ["resnet18", "resnet50"]
    CONV_TYPES = ["Conv2_5d", "Conv3d", "ACSConv"]
    CONV_MODEL_NAMES = {
        "Conv2_5d": "2.5D",
        "Conv3d": "3D",
        "ACSConv": "acs"
    }
    script_path = "experiments/MedMNIST3D/train_and_eval_pytorch.py"
    for dataset in datasets['3d']:
        for model_name in MODEL_NAMES:
            for conv_type in CONV_TYPES:
                for model_index in MODEL_INDEX:
                    command = (
                        f"python {script_path} "
                        f"--data_flag {dataset} "
                        f"--model_path weights/weights_{dataset}/{model_name}_{CONV_MODEL_NAMES[conv_type]}_{model_index}.pth "
                        f"--model_flag {model_name} --conv {conv_type} --num_epochs 0 --download --run {model_index} --batch_size {BATCH_SIZE_3D_DEFAULT * BATCH_SIZE_EXPAND_FACTOR_3D} "
                        f"--gpu_ids 0"
                    )
                    if dataset in DATASET_SPECIFIC_RULES:
                        command += f" {DATASET_SPECIFIC_RULES[dataset]}"
                    
                    handle_dataset_cache(dataset)
                    print(f"Executing: {command}")
                    subprocess.run(command, shell=True, check=True)
                    handle_dataset_cache(dataset, post_execution=True)
                    print(f"--------------------------------")
    print(f"================================================")
if __name__ == "__main__":
    main()