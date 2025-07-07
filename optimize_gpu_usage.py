#!/usr/bin/env python3
"""
Script to find optimal SIIM model configurations for specific GPU memory targets.
"""

def main():
    print("🎯 SIIM Model GPU Optimization for 24GB GPU")
    print("=" * 60)
    
    # Actual measured memory usage per model (in GB) - smaller models
    memory_per_model = {
        'small': 0.048,
        'standard': 0.192,
        'large': 0.766,
        'xlarge': 3.06,
        'xxlarge': 6.89,
        'mega': 12.24
    }
    
    gpu_memory = 24.0  # 24GB GPU
    target_usage = 12.0  # Target 12GB usage
    
    print(f"🖥️  GPU Memory: {gpu_memory}GB")
    print(f"🎯 Target Usage: {target_usage}GB")
    print(f"💾 Headroom: {gpu_memory - target_usage}GB\n")
    
    print("📊 OPTIMAL CONFIGURATIONS FOR ~12GB TARGET:")
    print("=" * 55)
    print(f"{'Model Size':<12} {'Clients':<8} {'Total Memory':<15} {'Utilization':<15}")
    print("-" * 55)
    
    recommendations = []
    
    for model_size, memory_mb in memory_per_model.items():
        # Calculate max clients that fit in target memory
        max_clients = int(target_usage / memory_mb)
        
        if max_clients > 0:
            total_memory = max_clients * memory_mb
            utilization = (total_memory / target_usage) * 100
            
            print(f"{model_size:<12} {max_clients:<8} {total_memory:<11.2f}GB   {utilization:<11.1f}%")
            
            # Store good recommendations
            if 0.8 <= utilization/100 <= 1.0:  # 80-100% of target
                recommendations.append({
                    'model_size': model_size,
                    'clients': max_clients,
                    'memory': total_memory,
                    'utilization': utilization
                })
    
    print(f"\n🏆 BEST CONFIGURATIONS FOR ~12GB:")
    print("-" * 40)
    
    if recommendations:
        for rec in recommendations:
            print(f"✅ {rec['clients']} × {rec['model_size'].upper()} models = {rec['memory']:.2f}GB ({rec['utilization']:.1f}%)")
            print(f"   Command: python fbd_main_siim.py --model_size {rec['model_size']} --parallel --num_clients {rec['clients']}")
    else:
        # Find closest matches
        print("🔍 Closest matches to 12GB target:")
        best_under = None
        best_over = None
        
        for model_size, memory_mb in memory_per_model.items():
            clients = int(target_usage / memory_mb)
            if clients > 0:
                total = clients * memory_mb
                if total <= target_usage and (best_under is None or total > best_under[2]):
                    best_under = (model_size, clients, total)
                
                clients_over = clients + 1
                total_over = clients_over * memory_mb
                if total_over > target_usage and (best_over is None or total_over < best_over[2]):
                    best_over = (model_size, clients_over, total_over)
        
        if best_under:
            print(f"📉 Just under: {best_under[1]} × {best_under[0].upper()} = {best_under[2]:.2f}GB")
        if best_over:
            print(f"📈 Just over: {best_over[1]} × {best_over[0].upper()} = {best_over[2]:.2f}GB")
    
    print(f"\n🔥 SPECIFIC RECOMMENDATIONS FOR YOU:")
    print("-" * 45)
    
    # Large model calculations
    large_memory = memory_per_model['large']
    large_15_clients = 15 * large_memory
    large_16_clients = 16 * large_memory
    
    print(f"🎪 BEST BALANCE:")
    print(f"   15 × LARGE models = {large_15_clients:.2f}GB (fits comfortably)")
    print(f"   16 × LARGE models = {large_16_clients:.2f}GB (close to target)")
    print(f"   Command: python fbd_main_siim.py --model_size large --parallel")
    print(f"   (Note: Default is 6 clients, you might need to modify config)")
    
    # XXLarge model
    xxlarge_memory = memory_per_model['xxlarge']
    print(f"\n💪 AGGRESSIVE UTILIZATION:")
    print(f"   1 × XXLARGE model = {xxlarge_memory:.2f}GB (great fit!)")
    print(f"   2 × XXLARGE models = {2 * xxlarge_memory:.2f}GB (perfect fit!)")
    print(f"   Command: python fbd_main_siim.py --model_size xxlarge --num_clients 2 --parallel")
    
    # Mixed strategies
    print(f"\n⚡ HYBRID STRATEGIES:")
    
    # Try combinations
    for large_clients in range(1, 8):
        remaining_memory = target_usage - (large_clients * large_memory)
        
        # See what else fits
        for model_size, memory in memory_per_model.items():
            if model_size == 'large':
                continue
            
            extra_clients = int(remaining_memory / memory)
            if extra_clients > 0:
                total = (large_clients * large_memory) + (extra_clients * memory)
                if 11.0 <= total <= 13.0:  # Close to 12GB target
                    print(f"   {large_clients} × LARGE + {extra_clients} × {model_size.upper()} = {total:.2f}GB")
    
    print(f"\n📋 CONFIGURATION NOTES:")
    print("-" * 30)
    print("• Default FedBD uses 6 clients - you may need to modify config")
    print("• Use --eval_on_cpu to save GPU memory during evaluation")
    print("• Monitor actual GPU usage with nvidia-smi during training")
    print("• Start with 15 × LARGE models for best balance")
    
    print(f"\n🚀 RECOMMENDED COMMANDS:")
    print(f"python fbd_main_siim.py --model_size large --parallel --num_clients 15 --auto_approval")
    print(f"python fbd_main_siim.py --model_size xxlarge --parallel --num_clients 2 --auto_approval")
    print(f"python fbd_main_siim.py --model_size large --parallel --num_clients 15 --eval_on_cpu --auto_approval")

if __name__ == "__main__":
    main()