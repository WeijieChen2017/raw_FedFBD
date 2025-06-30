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

fold_0 = {
    "bloodmnist": [
        "resnet18",
        "resnet50"
    ],
    "breastmnist": [
        "resnet18",
        "resnet50"
    ],
    "chestmnist": [
        "resnet18",
        "resnet50"
    ],
}

fold_1 = {
    "dermamnist": [
        # "resnet18", 
        # "resnet50"
    ],
    "octmnist": [
        # "resnet18",
        "resnet50"
    ],
    "pneumoniamnist": [
        "resnet18",
        "resnet50"
    ],
}

fold_2 = {
    "organamnist": [
        "resnet18",
        "resnet50"
    ],
    "organcmnist": [
        "resnet18",
        "resnet50"
    ],
    "organsmnist": [
        "resnet18",
        "resnet50"
    ],
}

fold_3 = {
    "pathmnist": [
        "resnet18",
        "resnet50"
    ],
    "retinamnist": [
        "resnet18",
        "resnet50"
    ],
    "tissuemnist": [
        "resnet18",
        "resnet50"
    ],
}

DATASET_SPECIFIC_RULES = {
    "breastmnist": "--as_rgb",
    "octmnist": "--as_rgb",
    "organcmnist": "--as_rgb",
    "tissuemnist": "--as_rgb",
    "pneumoniamnist": "--as_rgb",
    "chestmnist": "--as_rgb",
    "organamnist": "--as_rgb",
    "organsmnist": "--as_rgb",
}

BATCH_SIZE_RULES = {
    "bloodmnist":{
        "resnet18": 128, # 33327MiB /  40960MiB
        "resnet50": 32,
    },
    "breastmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "chestmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "dermamnist":{
        "resnet18": 128,
        "resnet50": 32, # 39921MiB /  40960MiB
    },
    "octmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "pneumoniamnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "organamnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "organcmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "organsmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "pathmnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "retinamnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
    "tissuemnist":{
        "resnet18": 128,
        "resnet50": 32,
    },
}

MODEL_INDEX = [1]
RANDOM_SEEDS = {1: 42, 2: 426, 3: 729}
CACHE_DIR = "data_storage"
MEDMNIST_DIR = os.path.expanduser("/root/.medmnist")
BATCH_SIZE_2D_DEFAULT = 32 
BATCH_SIZE_EXPAND_FACTOR_2D = 4
BATCH_SIZE_2D = BATCH_SIZE_2D_DEFAULT * BATCH_SIZE_EXPAND_FACTOR_2D

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
    """Run MedMNIST 2D training for all datasets."""
    parser = argparse.ArgumentParser(description="Run MedMNIST 2D training for all datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Run for a specific dataset (optional).")
    parser.add_argument("--fold", type=int, default=None, help="Run for a specific fold (0-3, optional).")
    args = parser.parse_args()

    # Determine which datasets and models to run
    tasks_to_run = {}
    if args.dataset:
        if args.dataset in DATASETS_2D:
            tasks_to_run[args.dataset] = MODEL_NAMES
            print(f"Running 2D training for {args.dataset}")
        else:
            print(f"Error: Dataset '{args.dataset}' not found in 2D datasets.")
            return
    else:
        fold_map = {0: fold_0, 1: fold_1, 2: fold_2, 3: fold_3}
        if args.fold is not None and args.fold in fold_map:
            tasks_to_run = fold_map[args.fold]
            print(f"Running 2D training for fold {args.fold}: {tasks_to_run}")
        else:
            tasks_to_run = {dataset: MODEL_NAMES for dataset in DATASETS_2D}
            print(f"Running 2D training for all datasets: {tasks_to_run}")

    # Create vanilla_2d directory if it doesn't exist
    vanilla_dir = "col2_vanilla_2d_224"
    if not os.path.exists(vanilla_dir):
        os.makedirs(vanilla_dir)

    print(f"================================================")
    # 2D commands - only image size 28
    IMAGE_SIZE = 224
    MODEL_NAMES = ["resnet18", "resnet50"]
    script_path = "experiments/MedMNIST2D/train_and_eval_pytorch.py"
    
    for model_index in MODEL_INDEX:
        for dataset, models_to_run in tasks_to_run.items():
            for model_name in models_to_run:
                # Create output directory for this specific training
                output_dir = os.path.join(vanilla_dir, f"{dataset}/{model_name}/{IMAGE_SIZE}/{model_index}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                batch_size = BATCH_SIZE_RULES.get(dataset, {}).get(model_name, BATCH_SIZE_2D)
                
                command = (
                    f"python {script_path} "
                    f"--data_flag {dataset} "
                    # f"--model_path weights/weights_{dataset}/{model_name}_{IMAGE_SIZE}_{model_index}.pth "
                    f"--model_flag {model_name} --size {IMAGE_SIZE} --num_epochs 100 --download --run {model_index} --batch_size {batch_size} "
                    f"--gpu_ids 0 --output_root {output_dir} --seed {RANDOM_SEEDS[model_index]}"
                )
                if dataset in DATASET_SPECIFIC_RULES:
                    command += f" {DATASET_SPECIFIC_RULES[dataset]}"
                
                handle_dataset_cache(dataset)
                print(f"Executing: {command}")
                subprocess.run(command, shell=True, check=True)
                handle_dataset_cache(dataset, post_execution=True)
                print(f"Results saved to: {output_dir}")
                print(f"--------------------------------")
    
    print(f"================================================")
    print(f"All 2D training completed. Results saved in {vanilla_dir}/ directory.")

if __name__ == "__main__":
    main()