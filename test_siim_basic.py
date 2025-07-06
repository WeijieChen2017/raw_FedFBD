#!/usr/bin/env python3
"""
Basic test of SIIM INFO definition
"""

print("Testing SIIM INFO definition...")

# Define SIIM INFO locally
SIIM_INFO = {
    'siim': {
        'task': 'segmentation',
        'description': 'SIIM-ACR Pneumothorax Segmentation Challenge',
        'n_channels': 1,
        'label': {'0': 'background', '1': 'pneumothorax'},
        'license': 'SIIM'
    }
}

# Test the SIIM info access
experiment_name = 'siim'

if experiment_name in SIIM_INFO:
    info = SIIM_INFO[experiment_name]
    task = info['task']
    print(f"✓ SIIM experiment found: task={task}")
    print(f"  Description: {info['description']}")
    print(f"  Channels: {info['n_channels']}")
    print(f"  Labels: {info['label']}")
else:
    print("✗ SIIM not found in INFO")

print("\n🎉 Basic SIIM INFO test passed!")