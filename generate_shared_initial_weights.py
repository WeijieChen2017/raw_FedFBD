#!/usr/bin/env python3
"""
Script to generate shared initial weights for SIIM UNet models.
This ensures all clients start with identical weights for fair federated learning.
"""

import torch
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fbd_models_siim import get_siim_model

def generate_initial_weights(model_size="standard", seed=42):
    """Generate and save initial weights for a SIIM UNet model."""
    
    print(f"🔧 Generating initial weights for {model_size} SIIM UNet model")
    print(f"📌 Using seed: {seed}")
    
    # Set random seed for reproducible initialization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create model with specified size
    model = get_siim_model(
        architecture="unet",
        in_channels=1,
        out_channels=1,
        model_size=model_size
    )
    
    # Get model state dict
    initial_weights = model.state_dict()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Model has {total_params:,} parameters")
    
    # Save weights
    output_file = f"siim_unet_initial_weights_{model_size}.pth"
    torch.save(initial_weights, output_file)
    
    print(f"💾 Saved initial weights to: {output_file}")
    
    # Verify saved weights
    loaded_weights = torch.load(output_file)
    print(f"✅ Verified: {len(loaded_weights)} weight tensors saved")
    
    # Show first few layer names and shapes
    print(f"\n🔍 Sample weight tensors:")
    for i, (name, tensor) in enumerate(list(loaded_weights.items())[:5]):
        print(f"  {i+1}. {name}: {tuple(tensor.shape)}")
    
    return output_file

def generate_all_sizes():
    """Generate initial weights for all model sizes."""
    model_sizes = ["small", "standard", "large", "xlarge", "xxlarge", "mega"]
    
    print("🏭 Generating initial weights for all model sizes")
    print("=" * 60)
    
    generated_files = []
    
    for size in model_sizes:
        try:
            output_file = generate_initial_weights(size)
            generated_files.append(output_file)
            print()
        except Exception as e:
            print(f"❌ Error generating weights for {size}: {e}")
            print()
    
    print("📋 Summary of generated files:")
    for file in generated_files:
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"  ✅ {file} ({file_size:.1f} MB)")
    
    print(f"\n🎯 Usage:")
    print(f"   Load these weights in your training script to ensure all clients start identically")
    print(f"   Example: torch.load('siim_unet_initial_weights_standard.pth')")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate initial weights for SIIM UNet models")
    parser.add_argument("--model_size", type=str, choices=["small", "standard", "large", "xlarge", "xxlarge", "mega"], 
                        default="standard", help="Model size to generate weights for")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight generation")
    parser.add_argument("--all", action="store_true", help="Generate weights for all model sizes")
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_sizes()
    else:
        generate_initial_weights(args.model_size, args.seed)