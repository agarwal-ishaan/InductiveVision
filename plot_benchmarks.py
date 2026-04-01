import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = "benchmark_plots"
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Plotting Receptive Field Theory
def plot_receptive_field():
    ## Theoretical Receptive Field Sizes over Layers (for 28x28 input)
    layers = np.arange(1, 6)
    
    # ResNet18 (modified for 28x28 with 3x3 convs)
    # RF grows roughly by +2 each 3x3 layer (stride 1). 
    # With strides (e.g., downsampling), it grows faster but structurally local.
    resnet_rf = [3, 7, 15, 31, 63]  # Approximated structural RF
    
    # ViT (global from layer 1 due to self-attention over all patches)
    # Our image is 28x28 = 784 pixels.
    vit_rf = [28*28]*5
    
    plt.figure(figsize=(8, 5))
    plt.plot(layers, resnet_rf, marker='o', label='CNN (SignLanguageCNN) - Local -> Global', color='blue', linewidth=2)
    plt.plot(layers, vit_rf, marker='s', label='ViT (Self-Attention) - Immediately Global', color='orange', linewidth=2)
    
    plt.title('Theoretical Receptive Field vs. Network Depth\n(Inductive Biases Compared)', fontsize=14)
    plt.xlabel('Network Depth (Blocks/Layers)', fontsize=12)
    plt.ylabel('Receptive Field Size (Pixels^2)', fontsize=12)
    plt.axhline(y=28*28, color='red', linestyle='--', alpha=0.5, label='Total Image Area (28x28)')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_receptive_field.png", dpi=300)
    plt.close()

# 2. Plotting Empirical Benchmarks
def plot_benchmarks():
    try:
        with open('benchmark_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("benchmark_results.json not found. Run benchmark_models.py first.")
        return
        
    df = pd.DataFrame(data)
    
    # Accuracy vs Data Percentage
    plt.figure(figsize=(10, 6))
    for model in ['SignLanguageCNN', 'ViT']:
        for aug in [False, True]:
            subset = df[(df['model'] == model) & (df['aug'] == aug)]
            if subset.empty: continue
            subset = subset.sort_values('pct')
            
            label = f"{model} (Aug={'Yes' if aug else 'No'})"
            marker = 'o' if model == 'SignLanguageCNN' else 's'
            linestyle = '-' if not aug else '--'
            
            plt.plot(subset['pct'] * 100, subset['acc'], marker=marker, linestyle=linestyle, label=label, linewidth=2)
            
    plt.title('Model Test Accuracy vs. Dataset Size', fontsize=14)
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xscale('log')
    plt.xticks([1, 5, 10, 50, 100], ['1%', '5%', '10%', '50%', '100%'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_accuracy_vs_data.png", dpi=300)
    plt.close()
    
    # Loss vs Data Percentage
    plt.figure(figsize=(10, 6))
    for model in ['SignLanguageCNN', 'ViT']:
        for aug in [False, True]:
            subset = df[(df['model'] == model) & (df['aug'] == aug)]
            if subset.empty: continue
            subset = subset.sort_values('pct')
            
            label = f"{model} (Aug={'Yes' if aug else 'No'})"
            marker = 'o' if model == 'SignLanguageCNN' else 's'
            linestyle = '-' if not aug else '--'
            
            plt.plot(subset['pct'] * 100, subset['loss'], marker=marker, linestyle=linestyle, label=label, linewidth=2)
            
    plt.title('Model Test Loss vs. Dataset Size', fontsize=14)
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xscale('log')
    plt.xticks([1, 5, 10, 50, 100], ['1%', '5%', '10%', '50%', '100%'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_loss_vs_data.png", dpi=300)
    plt.close()

    # Train Accuracy vs Data Percentage
    plt.figure(figsize=(10, 6))
    for model in ['SignLanguageCNN', 'ViT']:
        for aug in [False, True]:
            subset = df[(df['model'] == model) & (df['aug'] == aug)]
            if subset.empty: continue
            subset = subset.sort_values('pct')
            
            label = f"{model} (Aug={'Yes' if aug else 'No'})"
            marker = 'o' if model == 'SignLanguageCNN' else 's'
            linestyle = '-' if not aug else '--'
            
            plt.plot(subset['pct'] * 100, subset['train_acc'], marker=marker, linestyle=linestyle, label=label, linewidth=2)
            
    plt.title('Model Train Accuracy vs. Dataset Size', fontsize=14)
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12)
    plt.ylabel('Train Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xscale('log')
    plt.xticks([1, 5, 10, 50, 100], ['1%', '5%', '10%', '50%', '100%'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_train_accuracy_vs_data.png", dpi=300)
    plt.close()
    
    # Train Loss vs Data Percentage
    plt.figure(figsize=(10, 6))
    for model in ['SignLanguageCNN', 'ViT']:
        for aug in [False, True]:
            subset = df[(df['model'] == model) & (df['aug'] == aug)]
            if subset.empty: continue
            subset = subset.sort_values('pct')
            
            label = f"{model} (Aug={'Yes' if aug else 'No'})"
            marker = 'o' if model == 'SignLanguageCNN' else 's'
            linestyle = '-' if not aug else '--'
            
            plt.plot(subset['pct'] * 100, subset['train_loss'], marker=marker, linestyle=linestyle, label=label, linewidth=2)
            
    plt.title('Model Train Loss vs. Dataset Size', fontsize=14)
    plt.xlabel('Percentage of Training Data Used (%)', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xscale('log')
    plt.xticks([1, 5, 10, 50, 100], ['1%', '5%', '10%', '50%', '100%'])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_train_loss_vs_data.png", dpi=300)
    plt.close()

    # Generate Tables
    print("\n--- PERFORMANCE SUMMARY TABLE ---")
    agg_df = df.sort_values(by=['pct', 'model', 'aug']).reset_index(drop=True)
    agg_df['pct'] = (agg_df['pct'] * 100).astype(int).astype(str) + '%'
    agg_df['time'] = agg_df['time'].round(1)
    agg_df['memory_mb'] = agg_df['memory_mb'].round(1)
    agg_df['train_acc'] = agg_df['train_acc'].round(2)
    agg_df['acc'] = agg_df['acc'].round(2)
    agg_df['params'] = agg_df['params'].apply(lambda x: f"{x:,}")
    
    # Drop loss to save space in the text table
    agg_df = agg_df.drop(columns=['loss', 'train_loss'])
    
    agg_df = agg_df.rename(columns={
        'pct': 'Dataset Split',
        'model': 'Model', 'aug': 'Aug', 
        'train_acc': 'Train Acc (%)',
        'acc': 'Test Acc (%)', 
        'params': 'Parameters',
        'time': 'Time (s)',
        'memory_mb': 'Peak Mem (MB)'
    })
    
    print(agg_df.to_markdown(index=False))
    # Save the markdown table
    with open(f"{OUTPUT_DIR}/table_summary.md", 'w') as f:
        f.write(agg_df.to_markdown(index=False))
        
    print(f"\nAll plots and tables saved to '{OUTPUT_DIR}'/")

if __name__ == "__main__":
    plot_receptive_field()
    plot_benchmarks()
