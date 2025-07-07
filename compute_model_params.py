#!/usr/bin/env python3
"""
Script to compute the number of parameters in SIIM UNet models.
"""

import torch
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fbd_models_siim import get_siim_model

def count_parameters(model):
    """Count the total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format number with commas and in millions."""
    if num >= 1_000_000:
        return f"{num:,} ({num/1_000_000:.2f}M)"
    elif num >= 1_000:
        return f"{num:,} ({num/1_000:.1f}K)"
    else:
        return f"{num:,}"

def analyze_model_layers(model, model_name):
    """Analyze and print layer-wise parameter counts."""
    print(f"\n=== {model_name} Layer Analysis ===")
    total = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        print(f"{name:50s} {param.shape} -> {format_number(param_count)}")
    
    print(f"{'='*50}")
    print(f"{'Total':50s} -> {format_number(total)}")
    return total

def compute_memory_usage(model):
    """Estimate memory usage of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Each parameter is typically float32 (4 bytes)
    param_memory_mb = (total_params * 4) / (1024 * 1024)
    
    # During training, we also need gradients (another copy)
    training_memory_mb = param_memory_mb * 2
    
    # Add optimizer states (Adam typically needs 2x parameters for momentum and variance)
    optimizer_memory_mb = param_memory_mb * 2
    
    total_training_mb = param_memory_mb + training_memory_mb + optimizer_memory_mb
    
    return param_memory_mb, training_memory_mb, optimizer_memory_mb, total_training_mb

def main():
    print("🔍 SIIM UNet Model Parameter Analysis")
    print("=" * 60)
    
    # Test both model sizes
    model_configs = [
        ("Small Model", "small"),
        ("Standard Model", "standard")
    ]
    
    results = {}
    
    for model_name, model_size in model_configs:
        print(f"\n📊 {model_name} (size='{model_size}')")
        print("-" * 40)
        
        # Create model
        model = get_siim_model(
            architecture="unet",
            in_channels=1,
            out_channels=1,
            model_size=model_size
        )
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        print(f"Total parameters:     {format_number(total_params)}")
        print(f"Trainable parameters: {format_number(trainable_params)}")
        
        # Memory analysis
        param_mb, training_mb, optimizer_mb, total_mb = compute_memory_usage(model)
        
        print(f"\n💾 Memory Usage Estimates:")
        print(f"Model parameters:     {param_mb:.1f} MB")
        print(f"Gradients:           {training_mb - param_mb:.1f} MB")
        print(f"Optimizer states:    {optimizer_mb:.1f} MB")
        print(f"Total training:      {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        
        # Store results
        results[model_size] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': total_mb
        }
        
        # Show top 10 largest layers
        print(f"\n🔧 Top 5 Largest Layers:")
        layer_sizes = []
        for name, param in model.named_parameters():
            layer_sizes.append((name, param.numel(), param.shape))
        
        # Sort by parameter count
        layer_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, count, shape) in enumerate(layer_sizes[:5]):
            print(f"  {i+1}. {name[-40:]:40s} {str(shape):20s} {format_number(count)}")
        
        # Detailed layer analysis (optional - uncomment for full details)
        # analyze_model_layers(model, model_name)
    
    # Comparison
    print(f"\n📈 COMPARISON")
    print("=" * 40)
    
    small_params = results['small']['total_params']
    standard_params = results['standard']['total_params']
    
    ratio = standard_params / small_params if small_params > 0 else 1
    
    print(f"Standard model is {ratio:.1f}x larger than small model")
    print(f"Parameter difference: {format_number(standard_params - small_params)}")
    
    small_memory = results['small']['memory_mb']
    standard_memory = results['standard']['memory_mb']
    memory_ratio = standard_memory / small_memory if small_memory > 0 else 1
    
    print(f"Standard model uses {memory_ratio:.1f}x more memory than small model")
    print(f"Memory difference: {standard_memory - small_memory:.1f} MB")
    
    # GPU memory context
    print(f"\n🖥️  GPU Memory Context:")
    print(f"For a 16GB GPU: Small model uses {small_memory/16384*100:.1f}% | Standard model uses {standard_memory/16384*100:.1f}%")
    print(f"For a 24GB GPU: Small model uses {small_memory/24576*100:.1f}% | Standard model uses {standard_memory/24576*100:.1f}%")
    
    # How many models can fit?
    print(f"\n🔢 Parallel Model Capacity:")
    gpu_16gb = 16384
    gpu_24gb = 24576
    
    small_capacity_16gb = int(gpu_16gb / small_memory)
    standard_capacity_16gb = int(gpu_16gb / standard_memory)
    small_capacity_24gb = int(gpu_24gb / small_memory)
    standard_capacity_24gb = int(gpu_24gb / standard_memory)
    
    print(f"16GB GPU can fit: {small_capacity_16gb} small models OR {standard_capacity_16gb} standard models")
    print(f"24GB GPU can fit: {small_capacity_24gb} small models OR {standard_capacity_24gb} standard models")
    
    # FedBD simulation context
    print(f"\n⚡ FedBD Simulation Context:")
    print(f"With 6 clients (typical FedBD setup):")
    print(f"  - Small models: ✅ Fits easily on 16GB+ GPUs")
    print(f"  - Standard models: {'✅' if standard_capacity_16gb >= 6 else '❌'} {'Fits' if standard_capacity_16gb >= 6 else 'May not fit'} on 16GB GPU")
    print(f"  - For parallel training of 6 clients: {6 * small_memory:.0f}MB (small) vs {6 * standard_memory:.0f}MB (standard)")
    
    # Model data transfer
    single_small_mb = small_params * 4 / (1024 * 1024)  # Just model weights, no gradients/optimizer
    single_standard_mb = standard_params * 4 / (1024 * 1024)
    
    print(f"\n📡 Model Weight Transfer Sizes:")
    print(f"Small model weights: {single_small_mb:.1f}MB")
    print(f"Standard model weights: {single_standard_mb:.1f}MB")
    print(f"Ratio: Standard is {single_standard_mb/single_small_mb:.1f}x larger for network transfer")
    
    # Check actual model features
    print(f"\n🔧 Model Architecture Details:")
    small_model = get_siim_model(model_size="small")
    standard_model = get_siim_model(model_size="standard")
    
    # Get first conv layer to see actual feature sizes
    for name, param in small_model.named_parameters():
        if 'conv.unit0.conv.weight' in name and 'model.0' in name:
            small_features = param.shape[0]
            break
    
    for name, param in standard_model.named_parameters():
        if 'conv.unit0.conv.weight' in name and 'model.0' in name:
            standard_features = param.shape[0]
            break
    
    print(f"Small model features:    {small_features}")
    print(f"Standard model features: {standard_features}")

if __name__ == "__main__":
    main()