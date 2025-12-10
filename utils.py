"""Utility functions for metrics and visualization"""
import torch

def compute_miou(pred, target, num_classes=50, ignore_index=255):
    """Compute mean Intersection over Union"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Remove ignore_index pixels
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:
            iou = (intersection / union).item()
            ious.append(iou)
    
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
