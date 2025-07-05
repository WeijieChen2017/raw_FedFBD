# SIIM Dataset Configuration for FBD Framework

SIIM_CONFIG = {
    "dataset_name": "SIIM-ACR Pneumothorax",
    "description": "3D medical image segmentation dataset for pneumothorax detection",
    "task_type": "segmentation",
    "num_classes": 1,  # Binary segmentation
    "input_channels": 1,
    "model_architecture": "unet",
    "data_format": "nifti",  # .nii.gz files
    "image_size": (512, 512),  # Original image size
    "roi_size": (128, 128, 32),  # Region of interest for training
    "normalization": {
        "method": "min_max",
        "min_value": -1024,  # HU units
        "max_value": 1976
    }
}