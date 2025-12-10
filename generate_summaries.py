"""Generate Clean Summary PNGs"""
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.5

def load_results():
    results = []
    for file in glob('results/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    return results

def generate_batch_summary(results, batch_size):
    """Generate clean summary for specific batch size"""
    batch_results = [r for r in results if r['batch_size'] == batch_size]
    
    if not batch_results:
        return
    
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(f'Batch Size {batch_size} Summary - Multi-Task Learning (ImageNet + COCO)', 
                 fontsize=14, weight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, top=0.94, bottom=0.05, left=0.05, right=0.98)
    
    # 1. Results Table (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('tight')
    ax1.axis('off')
    ax1.set_title('Performance Metrics', fontsize=11, weight='bold', pad=10)
    
    table_data = []
    for r in sorted(batch_results, key=lambda x: (x['model'], x['optimizer'])):
        table_data.append([
            r['model'],
            r['optimizer'],
            f"{r['final_val_acc']:.1f}%",
            f"{r.get('final_val_miou', 0)*100:.1f}%",
            f"{r.get('best_val_acc', r['final_val_acc']):.1f}%",
            f"{r.get('best_val_miou', r.get('final_val_miou', 0))*100:.1f}%",
            f"{r.get('convergence_epoch_acc', 'N/A')}",
            f"{r.get('convergence_epoch_miou', 'N/A')}"
        ])
    
    headers = ['Model', 'Optimizer', 'Final\nAcc', 'Final\nmIoU', 'Best\nAcc', 'Best\nmIoU', 'Conv\nAcc', 'Conv\nmIoU']
    table = ax1.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.12, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 2. Best Configurations
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    ax2.set_title('Best Configurations', fontsize=11, weight='bold', pad=10)
    
    best_acc = max(batch_results, key=lambda x: x['final_val_acc'])
    best_miou = max(batch_results, key=lambda x: x.get('final_val_miou', 0))
    fastest = min(batch_results, key=lambda x: x.get('convergence_epoch_acc', 999))
    
    best_text = f"""Best Accuracy: {best_acc['final_val_acc']:.2f}%
  {best_acc['model']} + {best_acc['optimizer']}
  Converged: Epoch {best_acc.get('convergence_epoch_acc', 'N/A')}

Best mIoU: {best_miou.get('final_val_miou', 0)*100:.2f}%
  {best_miou['model']} + {best_miou['optimizer']}
  Converged: Epoch {best_miou.get('convergence_epoch_miou', 'N/A')}

Fastest Convergence: Epoch {fastest.get('convergence_epoch_acc', 'N/A')}
  {fastest['model']} + {fastest['optimizer']}
  Final Acc: {fastest['final_val_acc']:.2f}%"""
    
    ax2.text(0.5, 0.5, best_text, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.3), family='monospace')
    
    # 3. Accuracy Convergence Curves
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title('Classification Accuracy Convergence', fontsize=10, weight='bold')
    for r in batch_results:
        epochs = [h['epoch'] for h in r['history']]
        val_acc = [h['val_acc'] for h in r['history']]
        ax3.plot(epochs, val_acc, marker='o', markersize=4, linewidth=1.5, 
                label=f"{r['model']}-{r['optimizer']}")
    ax3.axhline(y=85, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 85%')
    ax3.set_xlabel('Epoch', fontsize=9)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=9)
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    
    # 4. mIoU Convergence Curves
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_title('Segmentation mIoU Convergence', fontsize=10, weight='bold')
    for r in batch_results:
        epochs = [h['epoch'] for h in r['history']]
        val_miou = [h['val_miou']*100 for h in r['history']]
        ax4.plot(epochs, val_miou, marker='s', markersize=4, linewidth=1.5,
                label=f"{r['model']}-{r['optimizer']}")
    ax4.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 30%')
    ax4.set_xlabel('Epoch', fontsize=9)
    ax4.set_ylabel('Validation mIoU (%)', fontsize=9)
    ax4.legend(fontsize=8, loc='lower right')
    ax4.grid(True, alpha=0.2, linewidth=0.5)
    
    # 5. Optimizer Comparison - Accuracy
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('Final Accuracy by Optimizer', fontsize=10, weight='bold')
    optimizers = sorted(set(r['optimizer'] for r in batch_results))
    opt_accs = [np.mean([r['final_val_acc'] for r in batch_results if r['optimizer'] == opt]) for opt in optimizers]
    bars = ax5.bar(range(len(optimizers)), opt_accs, width=0.6, color='#2E86AB', alpha=0.7)
    ax5.set_xticks(range(len(optimizers)))
    ax5.set_xticklabels(optimizers, fontsize=9)
    ax5.set_ylabel('Accuracy (%)', fontsize=9)
    ax5.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    for i, v in enumerate(opt_accs):
        ax5.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)
    
    # 6. Optimizer Comparison - mIoU
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title('Final mIoU by Optimizer', fontsize=10, weight='bold')
    opt_mious = [np.mean([r.get('final_val_miou', 0)*100 for r in batch_results if r['optimizer'] == opt]) for opt in optimizers]
    bars = ax6.bar(range(len(optimizers)), opt_mious, width=0.6, color='#A23B72', alpha=0.7)
    ax6.set_xticks(range(len(optimizers)))
    ax6.set_xticklabels(optimizers, fontsize=9)
    ax6.set_ylabel('mIoU (%)', fontsize=9)
    ax6.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    for i, v in enumerate(opt_mious):
        ax6.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)
    
    # 7. Model Comparison
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_title('Model Performance', fontsize=10, weight='bold')
    models = sorted(set(r['model'] for r in batch_results))
    model_accs = [np.mean([r['final_val_acc'] for r in batch_results if r['model'] == m]) for m in models]
    model_mious = [np.mean([r.get('final_val_miou', 0)*100 for r in batch_results if r['model'] == m]) for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax7.bar(x - width/2, model_accs, width, label='Accuracy', color='#2E86AB', alpha=0.7)
    ax7.bar(x + width/2, model_mious, width, label='mIoU', color='#A23B72', alpha=0.7)
    ax7.set_xticks(x)
    ax7.set_xticklabels(models, fontsize=9)
    ax7.set_ylabel('Performance (%)', fontsize=9)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    
    # 8. Convergence Speed
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.set_title('Convergence Speed (Epochs)', fontsize=10, weight='bold')
    opt_conv_acc = [np.mean([r.get('convergence_epoch_acc', r['epochs']) for r in batch_results if r['optimizer'] == opt]) for opt in optimizers]
    opt_conv_miou = [np.mean([r.get('convergence_epoch_miou', r['epochs']) for r in batch_results if r['optimizer'] == opt]) for opt in optimizers]
    x = np.arange(len(optimizers))
    width = 0.35
    ax8.bar(x - width/2, opt_conv_acc, width, label='85% Acc', color='#2E86AB', alpha=0.7)
    ax8.bar(x + width/2, opt_conv_miou, width, label='30% mIoU', color='#A23B72', alpha=0.7)
    ax8.set_xticks(x)
    ax8.set_xticklabels(optimizers, fontsize=9)
    ax8.set_ylabel('Epochs', fontsize=9)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    
    plt.savefig(f'results/batch{batch_size}_summary.png', dpi=200, bbox_inches='tight')
    print(f"✓ Saved: results/batch{batch_size}_summary.png")
    plt.close()

def generate_overall_summary(results):
    """Generate clean overall summary"""
    # TABLE ONLY PNG
    fig1 = plt.figure(figsize=(20, 6))
    ax1 = fig1.add_subplot(111)
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data = []
    for r in sorted(results, key=lambda x: (x['batch_size'], x['model'], x['optimizer'])):
        table_data.append([
            r['batch_size'],
            r['model'],
            r['optimizer'],
            f"{r['final_val_acc']:.1f}%",
            f"{r.get('final_val_miou', 0)*100:.1f}%",
            f"{r.get('best_val_acc', r['final_val_acc']):.1f}%",
            f"{r.get('best_val_miou', r.get('final_val_miou', 0))*100:.1f}%",
            f"{r.get('convergence_epoch_acc', 'N/A')}",
            f"{r.get('convergence_epoch_miou', 'N/A')}"
        ])
    
    headers = ['BS', 'Model', 'Optimizer', 'Final\nAcc', 'Final\nmIoU', 'Best\nAcc', 'Best\nmIoU', 'Conv\nAcc', 'Conv\nmIoU']
    table = ax1.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                      colWidths=[0.06, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig('results/overall_summary_table.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: results/overall_summary_table.png")
    plt.close()
    
    # VISUALIZATIONS PNG
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3, top=0.98, bottom=0.05, left=0.05, right=0.98)
    
    # 1. Best Overall
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.axis('off')
    ax2.set_title('Best Overall', fontsize=10, weight='bold', pad=10)
    
    best_acc = max(results, key=lambda x: x['final_val_acc'])
    best_miou = max(results, key=lambda x: x.get('final_val_miou', 0))
    
    best_text = f"""Accuracy: {best_acc['final_val_acc']:.2f}%
{best_acc['model']}+{best_acc['optimizer']}
BS: {best_acc['batch_size']}

mIoU: {best_miou.get('final_val_miou', 0)*100:.2f}%
{best_miou['model']}+{best_miou['optimizer']}
BS: {best_miou['batch_size']}"""
    
    ax2.text(0.5, 0.5, best_text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.3), family='monospace')
    
    # 2. Optimizer Ranking
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title('Optimizer Ranking (Avg)', fontsize=10, weight='bold')
    optimizers = sorted(set(r['optimizer'] for r in results))
    opt_scores = [(opt, np.mean([r['final_val_acc'] for r in results if r['optimizer'] == opt])) for opt in optimizers]
    opt_scores.sort(key=lambda x: x[1], reverse=True)
    opts, scores = zip(*opt_scores)
    ax3.barh(range(len(opts)), scores, color='#2E86AB', alpha=0.7)
    ax3.set_yticks(range(len(opts)))
    ax3.set_yticklabels(opts, fontsize=9)
    ax3.set_xlabel('Avg Accuracy (%)', fontsize=9)
    ax3.grid(True, alpha=0.2, axis='x', linewidth=0.5)
    for i, v in enumerate(scores):
        ax3.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    # 3. Batch Size Impact
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_title('Batch Size Impact', fontsize=10, weight='bold')
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    bs_accs = [np.mean([r['final_val_acc'] for r in results if r['batch_size'] == bs]) for bs in batch_sizes]
    bs_mious = [np.mean([r.get('final_val_miou', 0)*100 for r in results if r['batch_size'] == bs]) for bs in batch_sizes]
    ax4.plot(batch_sizes, bs_accs, 'o-', linewidth=2, markersize=8, label='Accuracy', color='#2E86AB')
    ax4.plot(batch_sizes, bs_mious, 's-', linewidth=2, markersize=8, label='mIoU', color='#A23B72')
    ax4.set_xlabel('Batch Size', fontsize=9)
    ax4.set_ylabel('Performance (%)', fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2, linewidth=0.5)
    
    # 4. Model Comparison
    ax5 = fig.add_subplot(gs[0, 3])
    ax5.set_title('Model Comparison', fontsize=10, weight='bold')
    models = sorted(set(r['model'] for r in results))
    model_accs = [np.mean([r['final_val_acc'] for r in results if r['model'] == m]) for m in models]
    model_mious = [np.mean([r.get('final_val_miou', 0)*100 for r in results if r['model'] == m]) for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax5.bar(x - width/2, model_accs, width, label='Accuracy', color='#2E86AB', alpha=0.7)
    ax5.bar(x + width/2, model_mious, width, label='mIoU', color='#A23B72', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, fontsize=9)
    ax5.set_ylabel('Performance (%)', fontsize=9)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    
    # 5-8. Convergence curves for each batch size
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    for idx, bs in enumerate(batch_sizes):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_title(f'BS {bs} Convergence', fontsize=10, weight='bold')
        bs_results = [r for r in results if r['batch_size'] == bs]
        for r in bs_results:
            epochs = [h['epoch'] for h in r['history']]
            val_acc = [h['val_acc'] for h in r['history']]
            ax.plot(epochs, val_acc, marker='o', markersize=3, linewidth=1,
                   label=f"{r['model'][:3]}-{r['optimizer']}")
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Val Acc (%)', fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2, linewidth=0.5)
    
    plt.savefig('results/overall_summary_visualizations.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: results/overall_summary_visualizations.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("📊 GENERATING CLEAN SUMMARY REPORTS")
    print("="*70 + "\n")
    
    results = load_results()
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"Loaded {len(results)} experiments\n")
    
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    for bs in batch_sizes:
        print(f"Generating summary for batch size {bs}...")
        generate_batch_summary(results, bs)
    
    print("\nGenerating overall summary...")
    generate_overall_summary(results)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
