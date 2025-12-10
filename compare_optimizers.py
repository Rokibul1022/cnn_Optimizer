"""Comprehensive Optimizer Comparison Script"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def load_results():
    """Load all result JSON files"""
    results = {}
    for file in glob('results/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            key = f"{data['model']}_{data['optimizer']}_bs{data['batch_size']}"
            results[key] = data
    return results

def create_comparison_table(results):
    """Create comparison table"""
    print("\n" + "="*120)
    print("OPTIMIZER COMPARISON TABLE")
    print("="*120)
    print(f"{'Model':<12} {'Optimizer':<10} {'Batch':<6} {'Val Acc':<10} {'Val mIoU':<10} {'Best Acc':<10} {'Best mIoU':<10} {'Conv Acc':<10} {'Conv mIoU':<10}")
    print("-"*120)
    
    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['model']:<12} {r['optimizer']:<10} {r['batch_size']:<6} "
              f"{r['final_val_acc']:<10.2f} {r['final_val_miou']*100:<10.2f} "
              f"{r['best_val_acc']:<10.2f} {r['best_val_miou']*100:<10.2f} "
              f"{r['convergence_epoch_acc']:<10} {r['convergence_epoch_miou']:<10}")
    print("="*120)

def plot_batch_size_comparison(results):
    """Compare optimizers across different batch sizes"""
    batch_sizes = sorted(set(r['batch_size'] for r in results.values()))
    optimizers = sorted(set(r['optimizer'] for r in results.values()))
    
    for model in ['mobilenet', 'resnet18']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Batch Size Comparison - {model.upper()}', fontsize=14, weight='bold')
        
        for opt in optimizers:
            bs_data = []
            acc_data = []
            miou_data = []
            
            for bs in batch_sizes:
                key = f"{model}_{opt}_bs{bs}"
                if key in results:
                    bs_data.append(bs)
                    acc_data.append(results[key]['final_val_acc'])
                    miou_data.append(results[key]['final_val_miou']*100)
            
            if bs_data:
                axes[0, 0].plot(bs_data, acc_data, 'o-', label=opt, linewidth=2, markersize=8)
                axes[0, 1].plot(bs_data, miou_data, 's-', label=opt, linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Validation Accuracy (%)')
        axes[0, 0].set_title('Classification Accuracy vs Batch Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Validation mIoU (%)')
        axes[0, 1].set_title('Segmentation mIoU vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bar charts
        x = np.arange(len(batch_sizes))
        width = 0.15
        for i, opt in enumerate(optimizers):
            acc_vals = []
            miou_vals = []
            for bs in batch_sizes:
                key = f"{model}_{opt}_bs{bs}"
                if key in results:
                    acc_vals.append(results[key]['final_val_acc'])
                    miou_vals.append(results[key]['final_val_miou']*100)
                else:
                    acc_vals.append(0)
                    miou_vals.append(0)
            
            axes[1, 0].bar(x + i*width, acc_vals, width, label=opt)
            axes[1, 1].bar(x + i*width, miou_vals, width, label=opt)
        
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Validation Accuracy (%)')
        axes[1, 0].set_title('Accuracy Comparison (Bar Chart)')
        axes[1, 0].set_xticks(x + width * (len(optimizers)-1)/2)
        axes[1, 0].set_xticklabels(batch_sizes)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Validation mIoU (%)')
        axes[1, 1].set_title('mIoU Comparison (Bar Chart)')
        axes[1, 1].set_xticks(x + width * (len(optimizers)-1)/2)
        axes[1, 1].set_xticklabels(batch_sizes)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'results/batch_comparison_{model}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: results/batch_comparison_{model}.png")
        plt.close()

def plot_model_comparison(results):
    """Compare models across optimizers"""
    optimizers = sorted(set(r['optimizer'] for r in results.values()))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: MobileNetV3 vs ResNet18', fontsize=14, weight='bold')
    
    for bs in [8, 16, 32]:
        mobile_acc = []
        mobile_miou = []
        resnet_acc = []
        resnet_miou = []
        valid_opts = []
        
        for opt in optimizers:
            mobile_key = f"mobilenet_{opt}_bs{bs}"
            resnet_key = f"resnet18_{opt}_bs{bs}"
            
            if mobile_key in results and resnet_key in results:
                valid_opts.append(opt)
                mobile_acc.append(results[mobile_key]['final_val_acc'])
                mobile_miou.append(results[mobile_key]['final_val_miou']*100)
                resnet_acc.append(results[resnet_key]['final_val_acc'])
                resnet_miou.append(results[resnet_key]['final_val_miou']*100)
        
        if valid_opts:
            x = np.arange(len(valid_opts))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, mobile_acc, width, label=f'MobileNet (bs={bs})')
            axes[0, 0].bar(x + width/2, resnet_acc, width, label=f'ResNet18 (bs={bs})')
            
            axes[0, 1].bar(x - width/2, mobile_miou, width, label=f'MobileNet (bs={bs})')
            axes[0, 1].bar(x + width/2, resnet_miou, width, label=f'ResNet18 (bs={bs})')
    
    axes[0, 0].set_xlabel('Optimizer')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Classification Accuracy by Model')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(valid_opts)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].set_xlabel('Optimizer')
    axes[0, 1].set_ylabel('Validation mIoU (%)')
    axes[0, 1].set_title('Segmentation mIoU by Model')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(valid_opts)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Convergence comparison
    for model in ['mobilenet', 'resnet18']:
        conv_acc = []
        conv_miou = []
        model_opts = []
        
        for opt in optimizers:
            key = f"{model}_{opt}_bs16"
            if key in results:
                model_opts.append(opt)
                conv_acc.append(results[key]['convergence_epoch_acc'])
                conv_miou.append(results[key]['convergence_epoch_miou'])
        
        if model_opts:
            x = np.arange(len(model_opts))
            width = 0.35
            
            axes[1, 0].bar(x if model == 'mobilenet' else x + width, conv_acc, width, label=model)
            axes[1, 1].bar(x if model == 'mobilenet' else x + width, conv_miou, width, label=model)
    
    axes[1, 0].set_xlabel('Optimizer')
    axes[1, 0].set_ylabel('Convergence Epoch')
    axes[1, 0].set_title('Convergence Speed (85% Accuracy)')
    axes[1, 0].set_xticks(x + width/2)
    axes[1, 0].set_xticklabels(model_opts)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].set_xlabel('Optimizer')
    axes[1, 1].set_ylabel('Convergence Epoch')
    axes[1, 1].set_title('Convergence Speed (30% mIoU)')
    axes[1, 1].set_xticks(x + width/2)
    axes[1, 1].set_xticklabels(model_opts)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/model_comparison.png")
    plt.close()

def plot_optimizer_ranking(results):
    """Rank optimizers by performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Optimizer Performance Ranking', fontsize=14, weight='bold')
    
    # Aggregate scores
    opt_scores = {}
    for key, r in results.items():
        opt = r['optimizer']
        if opt not in opt_scores:
            opt_scores[opt] = {'acc': [], 'miou': []}
        opt_scores[opt]['acc'].append(r['final_val_acc'])
        opt_scores[opt]['miou'].append(r['final_val_miou']*100)
    
    # Average scores
    opt_names = []
    avg_acc = []
    avg_miou = []
    for opt in sorted(opt_scores.keys()):
        opt_names.append(opt)
        avg_acc.append(np.mean(opt_scores[opt]['acc']))
        avg_miou.append(np.mean(opt_scores[opt]['miou']))
    
    # Sort by accuracy
    sorted_idx = np.argsort(avg_acc)[::-1]
    axes[0].barh([opt_names[i] for i in sorted_idx], [avg_acc[i] for i in sorted_idx])
    axes[0].set_xlabel('Average Validation Accuracy (%)')
    axes[0].set_title('Optimizer Ranking by Accuracy')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Sort by mIoU
    sorted_idx = np.argsort(avg_miou)[::-1]
    axes[1].barh([opt_names[i] for i in sorted_idx], [avg_miou[i] for i in sorted_idx])
    axes[1].set_xlabel('Average Validation mIoU (%)')
    axes[1].set_title('Optimizer Ranking by mIoU')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/optimizer_ranking.png', dpi=150, bbox_inches='tight')
    print("Saved: results/optimizer_ranking.png")
    plt.close()

def main():
    results = load_results()
    
    if not results:
        print("No results found in results/ folder!")
        return
    
    print(f"\nFound {len(results)} result files")
    
    # Create comparison table
    create_comparison_table(results)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_batch_size_comparison(results)
    plot_model_comparison(results)
    plot_optimizer_ranking(results)
    
    print("\n" + "="*120)
    print("COMPARISON COMPLETE!")
    print("="*120)
    print("\nGenerated files:")
    print("  - results/batch_comparison_mobilenet.png")
    print("  - results/batch_comparison_resnet18.png")
    print("  - results/model_comparison.png")
    print("  - results/optimizer_ranking.png")
    print("="*120 + "\n")

if __name__ == '__main__':
    main()
