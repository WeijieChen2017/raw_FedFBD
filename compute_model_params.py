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
    
    # Test all model sizes
    model_configs = [
        ("Small Model", "small"),
        ("Standard Model", "standard"),
        ("Large Model", "large"),
        ("XLarge Model", "xlarge"),
        ("XXLarge Model", "xxlarge"),
        ("Mega Model", "mega")
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
    print("=" * 50)
    
    small_params = results['small']['total_params']
    small_memory = results['small']['memory_mb']
    
    print(f"{'Model':<10} {'Parameters':<15} {'Memory':<12} {'vs Small':<12}")
    print("-" * 50)
    for size in ['small', 'standard', 'large', 'xlarge', 'xxlarge', 'mega']:
        if size in results:
            params = results[size]['total_params']
            memory = results[size]['memory_mb']
            ratio = params / small_params if small_params > 0 else 1
            memory_ratio = memory / small_memory if small_memory > 0 else 1
            
            print(f"{size:<10} {format_number(params):<15} {memory:<8.0f}MB {ratio:<8.1f}x")
    
    # GPU memory context
    print(f"\n🖥️  GPU Memory Context (per model):")
    gpu_16gb = 16384
    gpu_24gb = 24576
    
    print(f"{'Model':<10} {'16GB Usage':<12} {'24GB Usage':<12} {'Max on 16GB':<12} {'Max on 24GB':<12}")
    print("-" * 60)
    
    for size in ['small', 'standard', 'large', 'xlarge', 'xxlarge', 'mega']:
        if size in results:
            memory = results[size]['memory_mb']
            usage_16gb = memory / gpu_16gb * 100
            usage_24gb = memory / gpu_24gb * 100
            capacity_16gb = int(gpu_16gb / memory)
            capacity_24gb = int(gpu_24gb / memory)
            
            print(f"{size:<10} {usage_16gb:<8.1f}%    {usage_24gb:<8.1f}%    {capacity_16gb:<8d}     {capacity_24gb:<8d}")
    
    # FedBD simulation context for 24GB GPU
    print(f"\n⚡ FedBD Simulation Context (6 clients parallel):")
    print(f"{'Model':<10} {'Total Memory':<15} {'24GB Fit?':<10} {'Recommendation'}")
    print("-" * 55)
    
    for size in ['small', 'standard', 'large', 'xlarge', 'xxlarge', 'mega']:
        if size in results:
            memory = results[size]['memory_mb']
            total_6_clients = 6 * memory
            fits_24gb = total_6_clients < gpu_24gb
            
            if fits_24gb and total_6_clients < gpu_24gb * 0.8:  # Leave 20% headroom
                recommendation = "✅ Recommended"
            elif fits_24gb:
                recommendation = "⚠️  Tight fit"
            else:
                recommendation = "❌ Too large"
                
            print(f"{size:<10} {total_6_clients:<11.0f}MB   {'✅' if fits_24gb else '❌':<8s}  {recommendation}")
    
    # Model data transfer
    print(f"\n📡 Model Weight Transfer Sizes:")
    print(f"{'Model':<10} {'Weight Size':<12} {'vs Small'}")
    print("-" * 30)
    
    small_weight_mb = small_params * 4 / (1024 * 1024)
    
    for size in ['small', 'standard', 'large', 'xlarge', 'xxlarge', 'mega']:
        if size in results:
            params = results[size]['total_params']
            weight_mb = params * 4 / (1024 * 1024)
            ratio = weight_mb / small_weight_mb if small_weight_mb > 0 else 1
            
            print(f"{size:<10} {weight_mb:<8.1f}MB   {ratio:<8.1f}x")
    
    # Check actual model features
    print(f"\n🔧 Model Architecture Details:")
    print(f"{'Model':<10} {'Features':<10} {'First Conv Shape'}")
    print("-" * 40)
    
    for size in ['small', 'standard', 'large', 'xlarge', 'xxlarge', 'mega']:
        try:
            model = get_siim_model(model_size=size)
            # Get first conv layer to see actual feature sizes
            for name, param in model.named_parameters():
                if 'conv.unit0.conv.weight' in name and 'model.0' in name:
                    features = param.shape[0]
                    shape_str = f"{param.shape}"
                    print(f"{size:<10} {features:<10} {shape_str}")
                    break
        except Exception as e:
            print(f"{size:<10} ERROR: {e}")
    
    print(f"\n💡 For 24GB GPU Recommendations:")
    print(f"   🔥 LARGE model (256 features) - Great balance, fits 6 parallel clients")
    print(f"   🚀 XLARGE model (512 features) - High performance, 3-4 parallel clients")
    print(f"   💪 XXLARGE model (768 features) - Aggressive utilization, 1-2 parallel clients")
    print(f"   🏆 MEGA model (1024 features) - Maximum single model performance")
    print(f"   ⚡ Use --parallel mode for multi-client training")
    print(f"   📊 Monitor GPU memory usage during training")
    
    # Add specific recommendations for 12GB target
    print(f"\n🎯 For ~12GB GPU Utilization Target:")
    if 'xxlarge' in results:
        xxlarge_memory = results['xxlarge']['memory_mb']
        clients_12gb = int(12000 / xxlarge_memory)
        print(f"   • XXLARGE: {clients_12gb} parallel clients = {clients_12gb * xxlarge_memory:.0f}MB")
    
    if 'large' in results:
        large_memory = results['large']['memory_mb']
        clients_12gb_large = int(12000 / large_memory)
        print(f"   • LARGE: {clients_12gb_large} parallel clients = {clients_12gb_large * large_memory:.0f}MB")
    
    print(f"   🎪 Best balance: 8-9 LARGE models or 1 XXLARGE model for ~12GB usage")

if __name__ == "__main__":
    main()