"""Comparison Tool: Analyze and Compare All Results"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def load_all_results():
    """Load all JSON results from results folder"""
    results = []
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print("❌ No results folder found. Run training first.")
        return []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                data['filename'] = filename
                results.append(data)
    
    return results

def generate_summary_table(results):
    """Generate summary comparison table"""
    table_data = []
    
    for r in results:
        table_data.append([
            r['model'],
            r['optimizer'],
            r['batch_size'],
            r['epochs'],
            f"{r['final_val_acc']:.2f}%",
            f"{r['final_val_miou']:.4f}",
            f"{r['avg_epoch_time']:.2f}s",
            f"{r['total_time']:.2f}s"
        ])
    
    headers = ['Model', 'Optimizer', 'Batch Size', 'Epochs', 'Val Acc', 'Val mIoU', 'Avg Time/Epoch', 'Total Time']
    
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*120)
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("="*120 + "\n")
    
    return table_data, headers

def plot_optimizer_comparison(results):
    """Compare optimizers across different configurations"""
    # Group by optimizer
    optimizer_groups = {}
    for r in results:
        opt = r['optimizer']
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(r)
    
    if len(optimizer_groups) < 2:
        print("⚠️  Need at least 2 different optimizers for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Final Accuracy Comparison
    optimizers = list(optimizer_groups.keys())
    accuracies = [np.mean([r['final_val_acc'] for r in optimizer_groups[opt]]) for opt in optimizers]
    
    axes[0, 0].bar(range(len(optimizers)), accuracies, color='steelblue')
    axes[0, 0].set_xticks(range(len(optimizers)))
    axes[0, 0].set_xticklabels(optimizers, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Average Final Validation Accuracy by Optimizer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Final mIoU Comparison
    mious = [np.mean([r['final_val_miou'] for r in optimizer_groups[opt]]) for opt in optimizers]
    
    axes[0, 1].bar(range(len(optimizers)), mious, color='coral')
    axes[0, 1].set_xticks(range(len(optimizers)))
    axes[0, 1].set_xticklabels(optimizers, rotation=45, ha='right')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Average Final Validation mIoU by Optimizer')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Time Comparison
    times = [np.mean([r['avg_epoch_time'] for r in optimizer_groups[opt]]) for opt in optimizers]
    
    axes[1, 0].bar(range(len(optimizers)), times, color='lightgreen')
    axes[1, 0].set_xticks(range(len(optimizers)))
    axes[1, 0].set_xticklabels(optimizers, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Average Time per Epoch by Optimizer')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Efficiency Score (Accuracy / Time)
    efficiency = [accuracies[i] / times[i] if times[i] > 0 else 0 for i in range(len(optimizers))]
    
    axes[1, 1].bar(range(len(optimizers)), efficiency, color='mediumpurple')
    axes[1, 1].set_xticks(range(len(optimizers)))
    axes[1, 1].set_xticklabels(optimizers, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Efficiency (Acc% / Time)')
    axes[1, 1].set_title('Training Efficiency by Optimizer')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/optimizer_comparison.png")
    plt.close()

def plot_batch_size_comparison(results):
    """Compare performance across batch sizes"""
    # Group by batch size
    batch_groups = {}
    for r in results:
        bs = r['batch_size']
        if bs not in batch_groups:
            batch_groups[bs] = []
        batch_groups[bs].append(r)
    
    if len(batch_groups) < 2:
        print("⚠️  Need at least 2 different batch sizes for comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    batch_sizes = sorted(batch_groups.keys())
    accuracies = [np.mean([r['final_val_acc'] for r in batch_groups[bs]]) for bs in batch_sizes]
    mious = [np.mean([r['final_val_miou'] for r in batch_groups[bs]]) for bs in batch_sizes]
    times = [np.mean([r['avg_epoch_time'] for r in batch_groups[bs]]) for bs in batch_sizes]
    
    # Accuracy vs Batch Size
    axes[0].plot(batch_sizes, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Accuracy vs Batch Size')
    axes[0].grid(True, alpha=0.3)
    
    # mIoU vs Batch Size
    axes[1].plot(batch_sizes, mious, 'o-', linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('mIoU vs Batch Size')
    axes[1].grid(True, alpha=0.3)
    
    # Time vs Batch Size
    axes[2].plot(batch_sizes, times, 'o-', linewidth=2, markersize=8, color='lightgreen')
    axes[2].set_xlabel('Batch Size')
    axes[2].set_ylabel('Time per Epoch (s)')
    axes[2].set_title('Training Time vs Batch Size')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/batch_size_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/batch_size_comparison.png")
    plt.close()

def plot_model_comparison(results):
    """Compare different models"""
    # Group by model
    model_groups = {}
    for r in results:
        model = r['model']
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(r)
    
    if len(model_groups) < 2:
        print("⚠️  Need at least 2 different models for comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = list(model_groups.keys())
    accuracies = [np.mean([r['final_val_acc'] for r in model_groups[m]]) for m in models]
    mious = [np.mean([r['final_val_miou'] for r in model_groups[m]]) for m in models]
    times = [np.mean([r['avg_epoch_time'] for r in model_groups[m]]) for m in models]
    
    x = range(len(models))
    
    # Accuracy
    axes[0].bar(x, accuracies, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # mIoU
    axes[1].bar(x, mious, color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Model mIoU Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Time
    axes[2].bar(x, times, color='lightgreen')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models)
    axes[2].set_ylabel('Time per Epoch (s)')
    axes[2].set_title('Model Training Time Comparison')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/model_comparison.png")
    plt.close()

def plot_convergence_curves(results):
    """Plot training convergence for all experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for r in results:
        label = f"{r['optimizer']}-{r['model']}-bs{r['batch_size']}"
        epochs = [h['epoch'] for h in r['history']]
        val_acc = [h['val_acc'] for h in r['history']]
        val_miou = [h['val_miou'] for h in r['history']]
        
        axes[0].plot(epochs, val_acc, marker='o', label=label, linewidth=2)
        axes[1].plot(epochs, val_miou, marker='o', label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Convergence: Classification Accuracy')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation mIoU')
    axes[1].set_title('Convergence: Segmentation mIoU')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/convergence_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/convergence_curves.png")
    plt.close()

def generate_best_configurations(results):
    """Find and display best configurations"""
    print("\n" + "="*70)
    print("🏆 BEST CONFIGURATIONS")
    print("="*70)
    
    # Best accuracy
    best_acc = max(results, key=lambda x: x['final_val_acc'])
    print(f"\n✨ Best Accuracy: {best_acc['final_val_acc']:.2f}%")
    print(f"   Model: {best_acc['model']}")
    print(f"   Optimizer: {best_acc['optimizer']}")
    print(f"   Batch Size: {best_acc['batch_size']}")
    
    # Best mIoU
    best_miou = max(results, key=lambda x: x['final_val_miou'])
    print(f"\n✨ Best mIoU: {best_miou['final_val_miou']:.4f}")
    print(f"   Model: {best_miou['model']}")
    print(f"   Optimizer: {best_miou['optimizer']}")
    print(f"   Batch Size: {best_miou['batch_size']}")
    
    # Fastest training
    fastest = min(results, key=lambda x: x['avg_epoch_time'])
    print(f"\n⚡ Fastest Training: {fastest['avg_epoch_time']:.2f}s/epoch")
    print(f"   Model: {fastest['model']}")
    print(f"   Optimizer: {fastest['optimizer']}")
    print(f"   Batch Size: {fastest['batch_size']}")
    
    # Best efficiency (accuracy per second)
    for r in results:
        r['efficiency'] = r['final_val_acc'] / r['avg_epoch_time']
    best_eff = max(results, key=lambda x: x['efficiency'])
    print(f"\n🎯 Best Efficiency: {best_eff['efficiency']:.2f} (Acc%/s)")
    print(f"   Model: {best_eff['model']}")
    print(f"   Optimizer: {best_eff['optimizer']}")
    print(f"   Batch Size: {best_eff['batch_size']}")
    print(f"   Accuracy: {best_eff['final_val_acc']:.2f}%")
    print(f"   Time: {best_eff['avg_epoch_time']:.2f}s/epoch")
    
    print("="*70 + "\n")

def save_summary_report(results, table_data, headers):
    """Save comprehensive text report"""
    with open('results/summary_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULTI-TASK LEARNING BENCHMARK REPORT\n")
        f.write("ImageNet Classification + COCO2017 Segmentation\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Experiments: {len(results)}\n")
        f.write(f"Models Tested: {len(set(r['model'] for r in results))}\n")
        f.write(f"Optimizers Tested: {len(set(r['optimizer'] for r in results))}\n")
        f.write(f"Batch Sizes Tested: {sorted(set(r['batch_size'] for r in results))}\n\n")
        
        f.write("="*70 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*70 + "\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write("\n\n")
        
        # Best configurations
        best_acc = max(results, key=lambda x: x['final_val_acc'])
        best_miou = max(results, key=lambda x: x['final_val_miou'])
        fastest = min(results, key=lambda x: x['avg_epoch_time'])
        
        f.write("="*70 + "\n")
        f.write("BEST CONFIGURATIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Best Accuracy: {best_acc['final_val_acc']:.2f}%\n")
        f.write(f"  Config: {best_acc['model']} + {best_acc['optimizer']} (BS={best_acc['batch_size']})\n\n")
        
        f.write(f"Best mIoU: {best_miou['final_val_miou']:.4f}\n")
        f.write(f"  Config: {best_miou['model']} + {best_miou['optimizer']} (BS={best_miou['batch_size']})\n\n")
        
        f.write(f"Fastest: {fastest['avg_epoch_time']:.2f}s/epoch\n")
        f.write(f"  Config: {fastest['model']} + {fastest['optimizer']} (BS={fastest['batch_size']})\n\n")
    
    print("✓ Saved: results/summary_report.txt")

def main():
    print("\n" + "="*70)
    print("📊 RESULTS COMPARISON AND ANALYSIS")
    print("="*70)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("\n❌ No results found. Please run training first using train_interactive.py")
        return
    
    print(f"\n✓ Loaded {len(results)} experiment results\n")
    
    # Generate summary table
    table_data, headers = generate_summary_table(results)
    
    # Generate all comparison plots
    print("📈 Generating comparison plots...")
    plot_optimizer_comparison(results)
    plot_batch_size_comparison(results)
    plot_model_comparison(results)
    plot_convergence_curves(results)
    
    # Best configurations
    generate_best_configurations(results)
    
    # Save report
    save_summary_report(results, table_data, headers)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print("Generated files in results/:")
    print("  - optimizer_comparison.png")
    print("  - batch_size_comparison.png")
    print("  - model_comparison.png")
    print("  - convergence_curves.png")
    print("  - summary_report.txt")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
