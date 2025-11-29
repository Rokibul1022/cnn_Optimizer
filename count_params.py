"""Count exact parameters in models"""
import torch
from models import MultiTaskMobileNetV3, MultiTaskResNet18

def count_parameters(model, name):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    # Encoder
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Encoder: {encoder_params:,} params")
    
    # Classification head
    cls_params = sum(p.numel() for p in model.cls_head.parameters())
    print(f"Classification Head: {cls_params:,} params")
    
    # Detailed breakdown
    for i, layer in enumerate(model.cls_head):
        if hasattr(layer, 'weight'):
            w_params = layer.weight.numel()
            b_params = layer.bias.numel() if layer.bias is not None else 0
            print(f"  Layer {i} ({layer.__class__.__name__}): {w_params:,} weights + {b_params:,} bias = {w_params + b_params:,}")
    
    # Segmentation head
    seg_params = sum(p.numel() for p in model.seg_head.parameters())
    print(f"\nSegmentation Head: {seg_params:,} params")
    
    # Detailed breakdown
    for i, layer in enumerate(model.seg_head):
        if hasattr(layer, 'weight'):
            w_params = layer.weight.numel()
            b_params = layer.bias.numel() if layer.bias is not None else 0
            print(f"  Layer {i} ({layer.__class__.__name__}): {w_params:,} weights + {b_params:,} bias = {w_params + b_params:,}")
    
    # Total
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*70}")
    print(f"Total: {total_params:,} params")
    print(f"{'='*70}\n")

# Create models
mobilenet = MultiTaskMobileNetV3(num_classes=100, num_seg_classes=21)
resnet = MultiTaskResNet18(num_classes=100, num_seg_classes=21)

# Count parameters
count_parameters(mobilenet, "MobileNetV3-Large Multi-Task Model")
count_parameters(resnet, "ResNet-18 Multi-Task Model")
