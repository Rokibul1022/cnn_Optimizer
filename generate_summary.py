"""Generate Summary PNG from Results"""
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_results():
    results = []
    for file in glob('results/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    return results

def generate_summary_png():
    results = load_results()
    
    if not results:
        print("No results found!")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Multi-Task Learning Benchmark Summary\nImageNet Classification + COCO Segmentation', 
                 fontsize=16, weight='bold')
    
    # 1. Summary Statistics (top)
    ax1 = plt.subplot(4, 1, 1)
    ax1.axis('off')
    summary_text = f"""Total Experiments: {len(results)}  |  Models: {len(set(r['model'] for r in results))}  |  Optimizers: {len(set(r['optimizer'] for r in results))}  |  Batch Sizes: {sorted(set(r['batch_size'] for r in results))}"""
    ax1.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 2. Results Table
    ax2 = plt.subplot(4, 1, 2)
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            r['model'],
            r['optimizer'],
            r['batch_size'],
            r['epochs'],
            f"{r['final_val_acc']:.2f}%",
            f"{r.get('final_val_miou', 0)*100:.2f}%",
            f"{r.get('best_val_acc', r['final_val_acc']):.2f}%",
            f"{r.get('best_val_miou', r.get('final_val_miou', 0))*100:.2f}%"
        ])
    
    headers = ['Model', 'Optimizer', 'BS', 'Epochs', 'Final Acc', 'Final mIoU', 'Best Acc', 'Best mIoU']
    
    table = ax2.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 3. Best Configurations
    ax3 = plt.subplot(4, 1, 3)
    ax3.axis('off')
    
    best_acc = max(results, key=lambda x: x['final_val_acc'])
    best_miou = max(results, key=lambda x: x.get('final_val_miou', 0))
    
    best_text = f"""BEST CONFIGURATIONS

Best Accuracy: {best_acc['final_val_acc']:.2f}%
    Model: {best_acc['model']}  |  Optimizer: {best_acc['optimizer']}  |  Batch Size: {best_acc['batch_size']}

Best mIoU: {best_miou.get('final_val_miou', 0)*100:.2f}%
    Model: {best_miou['model']}  |  Optimizer: {best_miou['optimizer']}  |  Batch Size: {best_miou['batch_size']}"""
    
    ax3.text(0.5, 0.5, best_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), family='monospace')
    
    # 4. Best per Model
    ax4 = plt.subplot(4, 1, 4)
    ax4.axis('off')
    
    mobilenet_results = [r for r in results if r['model'] == 'mobilenet']
    resnet_results = [r for r in results if r['model'] == 'resnet18']
    
    model_text = "BEST OPTIMIZER PER MODEL\n\n"
    
    if mobilenet_results:
        best_mobile = max(mobilenet_results, key=lambda x: x['final_val_acc'])
        model_text += f"MobileNetV3: {best_mobile['optimizer']}\n"
        model_text += f"    Accuracy: {best_mobile['final_val_acc']:.2f}%  |  mIoU: {best_mobile.get('final_val_miou', 0)*100:.2f}%\n\n"
    
    if resnet_results:
        best_resnet = max(resnet_results, key=lambda x: x['final_val_acc'])
        model_text += f"ResNet18: {best_resnet['optimizer']}\n"
        model_text += f"    Accuracy: {best_resnet['final_val_acc']:.2f}%  |  mIoU: {best_resnet.get('final_val_miou', 0)*100:.2f}%"
    
    ax4.text(0.5, 0.5, model_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), family='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/summary.png")
    plt.close()

if __name__ == '__main__':
    generate_summary_png()
