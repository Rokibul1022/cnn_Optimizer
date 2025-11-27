"""Utility functions for metrics and visualization"""
import torch

def compute_miou(pred, target, num_classes=21):
    """Compute mean Intersection over Union"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(1, num_classes):  # Skip background
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou.item())
    
    return sum(ious) / len(ious) if ious else 0.0

def print_results_table(results):
    """Print results in table format"""
    print("\n┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
    print("│   Optimizer     │  Cls Accuracy   │   Seg mIoU      │  Convergence    │")
    print("│                 │                 │                 │     Speed       │")
    print("├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
    
    for opt, metrics in results.items():
        acc = f"{metrics['accuracy']:.1f}%"
        miou = f"{metrics['miou']:.2f}"
        speed = metrics['speed']
        print(f"│ {opt:15s} │ {acc:15s} │ {miou:15s} │ {speed:15s} │")
    
    print("└─────────────────┴─────────────────┴─────────────────┴─────────────────┘\n")
