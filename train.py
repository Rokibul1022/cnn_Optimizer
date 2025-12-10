"""Training script for 2-task learning"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

from models import MultiTaskMobileNetV3, MultiTaskResNet18
from datasets import ImageNetDataset, COCOSegmentationDataset, get_transforms
from muon import Muon, MTMuon
from utils import compute_miou

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, cls_loader, seg_loader, optimizer, cls_criterion, seg_criterion, use_mtmuon=False):
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
        
        cls_imgs = cls_imgs.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)
        seg_imgs = seg_imgs.to(device, non_blocking=True)
        seg_masks = seg_masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        cls_out, _ = model(cls_imgs, task='classification')
        _, seg_out = model(seg_imgs, task='segmentation')
        
        cls_loss = cls_criterion(cls_out, cls_labels)
        seg_loss = seg_criterion(seg_out, seg_masks)
        
        if use_mtmuon:
            combined_loss = optimizer.get_combined_loss(torch.stack([cls_loss, seg_loss]))
        else:
            combined_loss = cls_loss + 3.0 * seg_loss
        
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
        _, predicted = cls_out.max(1)
        cls_total += cls_labels.size(0)
        cls_correct += predicted.eq(cls_labels).sum().item()
        
        seg_pred = seg_out.argmax(1)
        seg_miou = compute_miou(seg_pred, seg_masks, num_classes=50, ignore_index=255)
        seg_ious.append(seg_miou)
        
        pbar.set_postfix({'loss': f'{combined_loss.item():.4f}', 
                         'acc': f'{100.*cls_correct/cls_total:.2f}%',
                         'mIoU': f'{seg_miou:.3f}'})
    
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
            
            cls_imgs = cls_imgs.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)
            seg_imgs = seg_imgs.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, non_blocking=True)
            
            cls_out, _ = model(cls_imgs, task='classification')
            _, seg_out = model(seg_imgs, task='segmentation')
            
            cls_loss = cls_criterion(cls_out, cls_labels)
            seg_loss = seg_criterion(seg_out, seg_masks)
            
            total_loss += (cls_loss + 3.0 * seg_loss).item()
            
            _, predicted = cls_out.max(1)
            cls_total += cls_labels.size(0)
            cls_correct += predicted.eq(cls_labels).sum().item()
            
            seg_pred = seg_out.argmax(1)
            seg_miou = compute_miou(seg_pred, seg_masks, num_classes=50, ignore_index=255)
            seg_ious.append(seg_miou)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': 100. * cls_correct / cls_total,
        'miou': sum(seg_ious) / len(seg_ious) if seg_ious else 0
    }

def interactive_menu():
    print("\n" + "="*70)
    print("   2-TASK MULTI-TASK LEARNING - OPTIMIZER COMPARISON")
    print("="*70)
    print("\nSelect Model:")
    print("  1. MobileNetV3")
    print("  2. ResNet18")
    print("="*70)
    
    model_choice = input("\nEnter model choice (1-2): ").strip()
    if model_choice == '1':
        model = 'mobilenet'
    elif model_choice == '2':
        model = 'resnet18'
    else:
        print("Invalid choice. Using MobileNetV3.")
        model = 'mobilenet'
    
    print("\n" + "="*70)
    print("\nSelect Optimizer to Train:")
    print("  1. SGD (Stochastic Gradient Descent)")
    print("  2. RMSprop")
    print("  3. Adam")
    print("  4. Muon (Momentum-based)")
    print("  5. MTMuon (Multi-Task Muon with Nash Equilibrium)")
    print("  6. Train All Optimizers")
    print("  0. Exit")
    print("="*70)
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == '0':
        print("Exiting...")
        return None, None, None, None
    elif choice in ['1', '2', '3', '4', '5', '6']:
        batch_size = input("Enter batch size (default 16): ").strip()
        batch_size = int(batch_size) if batch_size else 16
        
        epochs = input("Enter number of epochs (default 5): ").strip()
        epochs = int(epochs) if epochs else 5
        
        if choice == '1':
            return ['sgd'], batch_size, epochs, model
        elif choice == '2':
            return ['rmsprop'], batch_size, epochs, model
        elif choice == '3':
            return ['adam'], batch_size, epochs, model
        elif choice == '4':
            return ['muon'], batch_size, epochs, model
        elif choice == '5':
            return ['mtmuon'], batch_size, epochs, model
        elif choice == '6':
            return ['sgd', 'rmsprop', 'adam', 'muon', 'mtmuon'], batch_size, epochs, model
    else:
        print("Invalid choice. Please try again.")
        return interactive_menu()

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'rmsprop', 'adam', 'muon', 'mtmuon'])
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'resnet18'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--interactive', action='store_true', help='Use interactive menu')
    args = parser.parse_args()
    
    if args.optimizer is None or args.interactive:
        optimizers, batch_size, epochs, model = interactive_menu()
        if optimizers is None:
            return
        args.batch_size = batch_size
        args.epochs = epochs
        args.model = model
    else:
        optimizers = [args.optimizer]
    
    for opt in optimizers:
        train_single_optimizer(opt, args)

def train_single_optimizer(optimizer_name, args):
    print(f"\n{'='*70}")
    print(f"2-Task Learning: ImageNet Classification + COCO Segmentation")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    print("Loading datasets...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    imagenet_train = ImageNetDataset('datasets/imageNet', 'train', train_transform, subset_size=20000)
    imagenet_val = ImageNetDataset('datasets/imageNet', 'val', val_transform, subset_size=4000)
    coco_train = COCOSegmentationDataset('datasets/coco 2017', 'train', train_transform, subset_size=20000)
    coco_val = COCOSegmentationDataset('datasets/coco 2017', 'val', val_transform, subset_size=4000)
    
    cls_train_loader = DataLoader(imagenet_train, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=2, pin_memory=True)
    cls_val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=False, 
                                num_workers=2, pin_memory=True)
    seg_train_loader = DataLoader(coco_train, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=2, pin_memory=True)
    seg_val_loader = DataLoader(coco_val, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=2, pin_memory=True)
    
    if args.model == 'mobilenet':
        model = MultiTaskMobileNetV3(num_classes=100, num_seg_classes=50).to(device)
    else:
        model = MultiTaskResNet18(num_classes=100, num_seg_classes=50).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    elif optimizer_name == 'muon':
        optimizer = Muon(model.parameters(), lr=0.02)
    elif optimizer_name == 'mtmuon':
        optimizer = MTMuon(model.parameters(), lr=0.02, num_tasks=2)
    
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    print("Starting training...\n")
    results = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        train_metrics = train_epoch(model, cls_train_loader, seg_train_loader,
                                    optimizer, cls_criterion, seg_criterion, 
                                    use_mtmuon=(optimizer_name=='mtmuon'))
        val_metrics = validate(model, cls_val_loader, seg_val_loader, cls_criterion, seg_criterion)
        
        epoch_time = time.time() - start_time
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, mIoU: {train_metrics['miou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, mIoU: {val_metrics['miou']:.4f}")
        print(f"Time: {epoch_time:.2f}s\n")
        
        results.append({
            'epoch': epoch+1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_miou': train_metrics['miou'],
            'val_miou': val_metrics['miou']
        })
    
    os.makedirs('results', exist_ok=True)
    result_data = {
        'model': args.model,
        'optimizer': optimizer_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'final_train_loss': results[-1]['train_loss'],
        'final_val_loss': results[-1]['val_loss'],
        'final_train_acc': results[-1]['train_acc'],
        'final_val_acc': results[-1]['val_acc'],
        'final_train_miou': results[-1]['train_miou'],
        'final_val_miou': results[-1]['val_miou'],
        'best_val_acc': max(r['val_acc'] for r in results),
        'best_val_miou': max(r['val_miou'] for r in results),
        'convergence_epoch_acc': next((i+1 for i, r in enumerate(results) if r['val_acc'] >= 85), args.epochs),
        'convergence_epoch_miou': next((i+1 for i, r in enumerate(results) if r['val_miou'] >= 0.30), args.epochs),
        'history': results
    }
    
    result_file = f'results/{args.model}_{optimizer_name}_bs{args.batch_size}.json'
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Plot training history
    epochs = [r['epoch'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{args.model.upper()} - {optimizer_name.upper()} (bs={args.batch_size})', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(epochs, [r['train_loss'] for r in results], 'b-', label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, [r['val_loss'] for r in results], 'r-', label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, [r['train_acc'] for r in results], 'b-', label='Train Acc', marker='o')
    axes[0, 1].plot(epochs, [r['val_acc'] for r in results], 'r-', label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mIoU plot
    axes[1, 0].plot(epochs, [r['train_miou']*100 for r in results], 'b-', label='Train mIoU', marker='o')
    axes[1, 0].plot(epochs, [r['val_miou']*100 for r in results], 'r-', label='Val mIoU', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU (%)')
    axes[1, 0].set_title('Segmentation mIoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary table
    axes[1, 1].axis('off')
    summary_text = f"""Final Results:
    
Classification:
  Train Acc: {results[-1]['train_acc']:.2f}%
  Val Acc: {results[-1]['val_acc']:.2f}%
  Best Val: {result_data['best_val_acc']:.2f}%

Segmentation:
  Train mIoU: {results[-1]['train_miou']*100:.2f}%
  Val mIoU: {results[-1]['val_miou']*100:.2f}%
  Best Val: {result_data['best_val_miou']*100:.2f}%

Convergence:
  85% Acc: Epoch {result_data['convergence_epoch_acc']}
  30% mIoU: Epoch {result_data['convergence_epoch_miou']}

Hyperparameters:
  LR: {args.lr}
  Batch Size: {args.batch_size}
  Epochs: {args.epochs}"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plot_file = f'results/{args.model}_{optimizer_name}_bs{args.batch_size}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Val Acc: {results[-1]['val_acc']:.2f}%")
    print(f"Final Val mIoU: {results[-1]['val_miou']:.4f}")
    print(f"Best Val Acc: {result_data['best_val_acc']:.2f}%")
    print(f"Best Val mIoU: {result_data['best_val_miou']:.4f}")
    print(f"Convergence (85% acc): Epoch {result_data['convergence_epoch_acc']}")
    print(f"Convergence (30% mIoU): Epoch {result_data['convergence_epoch_miou']}")
    print(f"Results saved to: {result_file}")
    print(f"Plot saved to: {plot_file}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
