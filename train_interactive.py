"""Interactive Training Script with Results Tracking"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import MultiTaskMobileNetV3, MultiTaskResNet18
from datasets import ImageNetDataset, COCOSegmentationDataset, get_transforms
from mtadam_v2 import MTAdamV2
from utils import compute_miou

device = torch.device('mps' if torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')

OPTIMIZERS = {
    '1': ('SGD', 'sgd'),
    '2': ('SGD+Nesterov', 'sgd_nesterov'),
    '3': ('Adam', 'adam'),
    '4': ('AdamW', 'adamw'),
    '5': ('RMSprop', 'rmsprop'),
    '6': ('Adagrad', 'adagrad'),
    '7': ('Adadelta', 'adadelta'),
    '8': ('MTAdamV2', 'mtadamv2')
}

def train_epoch(model, cls_loader, seg_loader, optimizer, cls_criterion, seg_criterion, use_mtadam=False):
    model.train()
    total_loss = 0
    cls_correct = 0
    cls_total = 0
    seg_ious = []
    
    cls_iter = iter(cls_loader)
    seg_iter = iter(seg_loader)
    num_batches = min(len(cls_loader), len(seg_loader))
    
    pbar = tqdm(range(num_batches), desc='Training')
    for _ in pbar:
        try:
            cls_imgs, cls_labels = next(cls_iter)
            seg_imgs, seg_masks = next(seg_iter)
        except StopIteration:
            break
        
        cls_imgs, cls_labels = cls_imgs.to(device), cls_labels.to(device)
        seg_imgs, seg_masks = seg_imgs.to(device), seg_masks.to(device)
        
        optimizer.zero_grad()
        
        cls_out, _ = model(cls_imgs, task='classification')
        _, seg_out = model(seg_imgs, task='segmentation')
        
        cls_loss = cls_criterion(cls_out, cls_labels)
        seg_loss = seg_criterion(seg_out, seg_masks)
        
        if use_mtadam:
            combined_loss = optimizer.get_combined_loss(torch.stack([cls_loss, seg_loss]))
        else:
            combined_loss = cls_loss + seg_loss
        
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
        _, predicted = cls_out.max(1)
        cls_total += cls_labels.size(0)
        cls_correct += predicted.eq(cls_labels).sum().item()
        
        seg_pred = seg_out.argmax(1)
        miou = compute_miou(seg_pred, seg_masks, num_classes=21)
        seg_ious.append(miou)
        
        pbar.set_postfix({'loss': f'{combined_loss.item():.4f}', 
                         'acc': f'{100.*cls_correct/cls_total:.2f}%',
                         'mIoU': f'{miou:.4f}'})
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': 100. * cls_correct / cls_total,
        'miou': sum(seg_ious) / len(seg_ious) if seg_ious else 0
    }

def validate(model, cls_loader, seg_loader, cls_criterion, seg_criterion):
    model.eval()
    total_loss = 0
    cls_correct = 0
    cls_total = 0
    seg_ious = []
    
    with torch.no_grad():
        cls_iter = iter(cls_loader)
        seg_iter = iter(seg_loader)
        num_batches = min(len(cls_loader), len(seg_loader))
        
        for _ in range(num_batches):
            try:
                cls_imgs, cls_labels = next(cls_iter)
                seg_imgs, seg_masks = next(seg_iter)
            except StopIteration:
                break
            
            cls_imgs, cls_labels = cls_imgs.to(device), cls_labels.to(device)
            seg_imgs, seg_masks = seg_imgs.to(device), seg_masks.to(device)
            
            cls_out, _ = model(cls_imgs, task='classification')
            _, seg_out = model(seg_imgs, task='segmentation')
            
            cls_loss = cls_criterion(cls_out, cls_labels)
            seg_loss = seg_criterion(seg_out, seg_masks)
            
            total_loss += (cls_loss + seg_loss).item()
            
            _, predicted = cls_out.max(1)
            cls_total += cls_labels.size(0)
            cls_correct += predicted.eq(cls_labels).sum().item()
            
            seg_pred = seg_out.argmax(1)
            miou = compute_miou(seg_pred, seg_masks, num_classes=21)
            seg_ious.append(miou)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': 100. * cls_correct / cls_total,
        'miou': sum(seg_ious) / len(seg_ious) if seg_ious else 0
    }

def save_results(results, model_name, optimizer_name, batch_size):
    """Save results to JSON and generate plots"""
    os.makedirs('results', exist_ok=True)
    
    filename = f"{model_name}_{optimizer_name}_bs{batch_size}"
    
    # Save JSON
    with open(f'results/{filename}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    epochs = [r['epoch'] for r in results['history']]
    train_acc = [r['train_acc'] for r in results['history']]
    val_acc = [r['val_acc'] for r in results['history']]
    train_miou = [r['train_miou'] for r in results['history']]
    val_miou = [r['val_miou'] for r in results['history']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(epochs, train_acc, 'b-', label='Train')
    axes[0].plot(epochs, val_acc, 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Classification Accuracy\n{model_name} - {optimizer_name}')
    axes[0].legend()
    axes[0].grid(True)
    
    # mIoU plot
    axes[1].plot(epochs, train_miou, 'b-', label='Train')
    axes[1].plot(epochs, val_miou, 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title(f'Segmentation mIoU\n{model_name} - {optimizer_name}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Results saved to results/{filename}.json")
    print(f"✓ Plots saved to results/{filename}.png")

def main():
    print("\n" + "="*70)
    print("Multi-Task Learning: ImageNet + COCO2017")
    print("Interactive Training System")
    print("="*70)
    
    # Select Model
    print("\n📦 SELECT MODEL:")
    print("1. MobileNetV3-Large (6.1M params, faster)")
    print("2. ResNet-18 (12.9M params, more accurate)")
    model_choice = input("Enter choice (1 or 2): ").strip()
    
    if model_choice == '1':
        model_name = 'mobilenetv3'
        model_class = MultiTaskMobileNetV3
    elif model_choice == '2':
        model_name = 'resnet18'
        model_class = MultiTaskResNet18
    else:
        print("Invalid choice. Using MobileNetV3.")
        model_name = 'mobilenetv3'
        model_class = MultiTaskMobileNetV3
    
    # Select Batch Size
    print("\n📊 SELECT BATCH SIZE:")
    print("Recommended: 4, 8, 16, 32")
    batch_size = int(input("Enter batch size: ").strip())
    
    # Select Optimizer
    print("\n⚙️  SELECT OPTIMIZER:")
    for key, (name, _) in OPTIMIZERS.items():
        print(f"{key}. {name}")
    opt_choice = input("Enter choice (1-8): ").strip()
    
    if opt_choice not in OPTIMIZERS:
        print("Invalid choice. Using Adam.")
        opt_choice = '3'
    
    optimizer_display, optimizer_name = OPTIMIZERS[opt_choice]
    
    # Select Epochs
    print("\n🔄 SELECT EPOCHS:")
    epochs = int(input("Enter number of epochs: ").strip())
    
    # Learning Rate
    print("\n📈 SELECT LEARNING RATE:")
    lr = float(input("Enter learning rate (default 0.001): ").strip() or "0.001")
    
    print("\n" + "="*70)
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Optimizer: {optimizer_display}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Device: {device}")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    imagenet_train = ImageNetDataset('datasets/imageNet', 'train', train_transform, subset_size=50000)
    imagenet_val = ImageNetDataset('datasets/imageNet', 'val', val_transform, subset_size=10000)
    coco_train = COCOSegmentationDataset('datasets/coco 2017', 'train', train_transform, subset_size=50000)
    coco_val = COCOSegmentationDataset('datasets/coco 2017', 'val', val_transform, subset_size=5000)
    
    cls_train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, num_workers=2)
    cls_val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False, num_workers=2)
    seg_train_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=True, num_workers=2)
    seg_val_loader = DataLoader(coco_val, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print(f"\n🏗️  Creating {model_name} model...")
    model = model_class(num_classes=1000, num_seg_classes=21).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'sgd_nesterov':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == 'mtadamv2':
        optimizer = MTAdamV2(model.parameters(), lr=lr, num_tasks=2)
    
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("\n🚀 Starting training...\n")
    history = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        train_metrics = train_epoch(model, cls_train_loader, seg_train_loader, 
                                    optimizer, cls_criterion, seg_criterion, 
                                    use_mtadam=(optimizer_name=='mtadamv2'))
        val_metrics = validate(model, cls_val_loader, seg_val_loader, cls_criterion, seg_criterion)
        
        epoch_time = time.time() - start_time
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, mIoU: {train_metrics['miou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, mIoU: {val_metrics['miou']:.4f}")
        print(f"Time: {epoch_time:.2f}s\n")
        
        history.append({
            'epoch': epoch+1,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_miou': train_metrics['miou'],
            'val_miou': val_metrics['miou'],
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'time': epoch_time
        })
    
    # Save results
    results = {
        'model': model_name,
        'optimizer': optimizer_display,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': lr,
        'final_val_acc': history[-1]['val_acc'],
        'final_val_miou': history[-1]['val_miou'],
        'avg_epoch_time': sum(h['time'] for h in history) / len(history),
        'total_time': sum(h['time'] for h in history),
        'history': history
    }
    
    save_results(results, model_name, optimizer_name, batch_size)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Final Validation Accuracy: {results['final_val_acc']:.2f}%")
    print(f"Final Validation mIoU: {results['final_val_miou']:.4f}")
    print(f"Average Time/Epoch: {results['avg_epoch_time']:.2f}s")
    print(f"Total Training Time: {results['total_time']:.2f}s")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
