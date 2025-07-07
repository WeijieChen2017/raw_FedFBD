#!/usr/bin/env python3
"""
Script to test and compare GPU vs CPU evaluation performance for SIIM models.
"""

import torch
import time
import sys
import os
import psutil
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fbd_models_siim import get_siim_model

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_cpu_memory():
    """Get current CPU memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def create_dummy_data(batch_size=4, roi_size=(128, 128, 32)):
    """Create dummy SIIM-like data for testing."""
    # Image data: batch_size x 1 x H x W x D
    images = torch.randn(batch_size, 1, *roi_size)
    # Label data: batch_size x 1 x H x W x D (binary segmentation)
    labels = torch.randint(0, 2, (batch_size, 1, *roi_size)).float()
    
    return {"image": images, "label": labels}

def test_model_evaluation(model, data_batch, device, num_iterations=10):
    """Test model evaluation speed and memory usage."""
    model.eval()
    model.to(device)
    
    # Move data to device
    inputs = data_batch["image"].to(device)
    labels = data_batch["label"].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(inputs)
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Measure initial memory
    initial_gpu_mem = get_gpu_memory()
    initial_cpu_mem = get_cpu_memory()
    
    # Timing test
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iterations):
            outputs = model(inputs)
            
            # Simulate loss calculation
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)
            
            # Simulate dice calculation
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid > 0.5).float()
            
            # Simple dice calculation
            intersection = (outputs_binary * labels).sum()
            dice = (2.0 * intersection) / (outputs_binary.sum() + labels.sum() + 1e-8)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Measure final memory
    final_gpu_mem = get_gpu_memory()
    final_cpu_mem = get_cpu_memory()
    
    avg_time = (end_time - start_time) / num_iterations
    gpu_mem_used = final_gpu_mem - initial_gpu_mem
    cpu_mem_used = final_cpu_mem - initial_cpu_mem
    
    return {
        'avg_time': avg_time,
        'gpu_memory_mb': final_gpu_mem,
        'cpu_memory_mb': final_cpu_mem,
        'gpu_mem_increase': gpu_mem_used,
        'cpu_mem_increase': cpu_mem_used
    }

def main():
    print("🔍 SIIM Model Evaluation Performance Test")
    print("=" * 60)
    
    # Test configurations
    model_sizes = ['small', 'standard', 'large']
    if torch.cuda.is_available():
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_total_memory:.1f}GB)")
    else:
        print("GPU: Not available")
    
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    
    # Create test data
    print(f"\nCreating test data (batch_size=4, roi_size=128x128x32)...")
    test_data = create_dummy_data(batch_size=4)
    
    results = {}
    
    for model_size in model_sizes:
        print(f"\n📊 Testing {model_size.upper()} Model")
        print("-" * 40)
        
        # Create model
        try:
            model = get_siim_model(model_size=model_size)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
            
            results[model_size] = {}
            
            for device in devices:
                device_name = "GPU" if device.type == 'cuda' else "CPU"
                print(f"\n{device_name} Evaluation:")
                
                try:
                    # Clear memory before test
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    result = test_model_evaluation(model, test_data, device, num_iterations=5)
                    
                    print(f"  Time per iteration: {result['avg_time']*1000:.1f}ms")
                    print(f"  Total GPU memory: {result['gpu_memory_mb']:.1f}MB")
                    print(f"  Total CPU memory: {result['cpu_memory_mb']:.1f}MB")
                    
                    results[model_size][device_name.lower()] = result
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    results[model_size][device_name.lower()] = None
                
                # Clean up
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"❌ Error creating {model_size} model: {e}")
    
    # Summary comparison
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Model':<10} {'Device':<6} {'Time/iter':<12} {'GPU Memory':<12} {'Speedup'}")
    print("-" * 60)
    
    for model_size in model_sizes:
        if model_size in results:
            cpu_result = results[model_size].get('cpu')
            gpu_result = results[model_size].get('gpu')
            
            if cpu_result:
                print(f"{model_size:<10} CPU    {cpu_result['avg_time']*1000:<8.1f}ms   {cpu_result['gpu_memory_mb']:<8.1f}MB   1.0x")
            
            if gpu_result and cpu_result:
                speedup = cpu_result['avg_time'] / gpu_result['avg_time']
                print(f"{model_size:<10} GPU    {gpu_result['avg_time']*1000:<8.1f}ms   {gpu_result['gpu_memory_mb']:<8.1f}MB   {speedup:<.1f}x")
            elif gpu_result:
                print(f"{model_size:<10} GPU    {gpu_result['avg_time']*1000:<8.1f}ms   {gpu_result['gpu_memory_mb']:<8.1f}MB   N/A")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print("For 24GB GPU:")
        print("✅ GPU evaluation recommended for all model sizes")
        print("⚡ GPU provides significant speedup (typically 5-20x)")
        print("📊 Use --eval_on_cpu only if GPU memory is needed for training")
        print("\nMemory-conscious options:")
        print("🔥 Large model + GPU eval: Good balance")
        print("💾 XLarge model + CPU eval: Maximum model size")
        print("⚖️  Large model + CPU eval: Conservative approach")
    else:
        print("CPU-only system:")
        print("📊 All evaluation will be on CPU")
        print("💡 Consider using smaller models for faster evaluation")
    
    print(f"\nUsage examples:")
    print(f"python fbd_main_siim.py --model_size large                    # GPU training + GPU eval")
    print(f"python fbd_main_siim.py --model_size large --eval_on_cpu      # GPU training + CPU eval")
    print(f"python fbd_main_siim.py --model_size xlarge --eval_on_cpu     # GPU training + CPU eval")

if __name__ == "__main__":
    main()