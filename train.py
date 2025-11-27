"""Training script for multi-task learning"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
from tqdm import tqdm

from models import MultiTaskMobileNetV3, MultiTaskResNet18
from datasets import ImageNetDataset, COCOSegmentationDataset, get_transforms
from mtadam_v2 import MTAdamV2
from utils import compute_miou, print_results_table

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # Forward
        cls_out, _ = model(cls_imgs, task='classification')
        _, seg_out = model(seg_imgs, task='segmentation')
        
        # Losses
        cls_loss = cls_criterion(cls_out, cls_labels)
        seg_loss = seg_criterion(seg_out, seg_masks)
        
        # Combine losses
        if use_mtadam:
            combined_loss = optimizer.get_combined_loss(torch.stack([cls_loss, seg_loss]))
        else:
            combined_loss = cls_loss + seg_loss
        
        combined_loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += combined_loss.item()
        _, predicted = cls_out.max(1)
        cls_total += cls_labels.size(0)
        cls_correct += predicted.eq(cls_labels).sum().item()
        
        # mIoU
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'sgd_nesterov', 'adam', 'adamw', 'rmsprop', 'adagrad', 'mtadamv2'])
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'resnet18'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Multi-Task Learning: ImageNet + COCO2017")
    print(f"{'='*70}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Datasets
    print("Loading datasets...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    imagenet_train = ImageNetDataset('datasets/imageNet', 'train', train_transform, subset_size=5000)
    imagenet_val = ImageNetDataset('datasets/imageNet', 'val', val_transform, subset_size=1000)
    coco_train = COCOSegmentationDataset('datasets/coco 2017', 'train', train_transform, subset_size=2000)
    coco_val = COCOSegmentationDataset('datasets/coco 2017', 'val', val_transform, subset_size=500)
    
    cls_train_loader = DataLoader(imagenet_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    cls_val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    seg_train_loader = DataLoader(coco_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    seg_val_loader = DataLoader(coco_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    if args.model == 'mobilenet':
        model = MultiTaskMobileNetV3(num_classes=100, num_seg_classes=21).to(device)
    else:
        model = MultiTaskResNet18(num_classes=100, num_seg_classes=21).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'sgd_nesterov':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'mtadamv2':
        optimizer = MTAdamV2(model.parameters(), lr=args.lr, num_tasks=2)
    
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Starting training...\n")
    results = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        train_metrics = train_epoch(model, cls_train_loader, seg_train_loader, 
                                    optimizer, cls_criterion, seg_criterion, 
                                    use_mtadam=(args.optimizer=='mtadamv2'))
        val_metrics = validate(model, cls_val_loader, seg_val_loader, cls_criterion, seg_criterion)
        
        epoch_time = time.time() - start_time
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, mIoU: {train_metrics['miou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, mIoU: {val_metrics['miou']:.4f}")
        print(f"Time: {epoch_time:.2f}s\n")
        
        results.append({
            'epoch': epoch+1,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_miou': train_metrics['miou'],
            'val_miou': val_metrics['miou'],
            'time': epoch_time
        })
    
    # Final results
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Validation Accuracy: {results[-1]['val_acc']:.2f}%")
    print(f"Final Validation mIoU: {results[-1]['val_miou']:.4f}")
    print(f"Average Time/Epoch: {sum(r['time'] for r in results)/len(results):.2f}s")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
