"""Training script for 3-task learning: Classification + COCO + Open Images"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import argparse
from tqdm import tqdm
import json

from models_3task import MultiTaskMobileNetV3
from datasets import ImageNetDataset, COCOSegmentationDataset, get_transforms
from openimages_dataset import OpenImagesSegmentationDataset
from muon import Muon, MTMuon
from utils import compute_miou

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, cls_loader, coco_loader, oi_loader, optimizer, cls_criterion, seg_criterion, use_mtmuon=False):
    model.train()
    total_loss = 0
    cls_correct = 0
    cls_total = 0
    coco_ious = []
    oi_ious = []
    
    cls_iter = iter(cls_loader)
    coco_iter = iter(coco_loader)
    oi_iter = iter(oi_loader)
    num_batches = min(len(cls_loader), len(coco_loader), len(oi_loader))
    
    pbar = tqdm(range(num_batches), desc='Training')
    for _ in pbar:
        try:
            cls_imgs, cls_labels = next(cls_iter)
            coco_imgs, coco_masks = next(coco_iter)
            oi_imgs, oi_masks = next(oi_iter)
        except StopIteration:
            break
        
        cls_imgs, cls_labels = cls_imgs.to(device), cls_labels.to(device)
        coco_imgs, coco_masks = coco_imgs.to(device), coco_masks.to(device)
        oi_imgs, oi_masks = oi_imgs.to(device), oi_masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        cls_out, _, _ = model(cls_imgs, task='classification')
        _, coco_out, _ = model(coco_imgs, task='segmentation_coco')
        _, _, oi_out = model(oi_imgs, task='segmentation_oi')
        
        # Losses
        cls_loss = cls_criterion(cls_out, cls_labels)
        coco_loss = seg_criterion(coco_out, coco_masks)
        oi_loss = seg_criterion(oi_out, oi_masks)
        
        # Combine losses
        if use_mtmuon:
            combined_loss = optimizer.get_combined_loss(torch.stack([cls_loss, coco_loss, oi_loss]))
        else:
            combined_loss = (cls_loss + coco_loss + oi_loss) / 3.0
        
        combined_loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += combined_loss.item()
        _, predicted = cls_out.max(1)
        cls_total += cls_labels.size(0)
        cls_correct += predicted.eq(cls_labels).sum().item()
        
        # mIoU
        coco_pred = coco_out.argmax(1)
        coco_miou = compute_miou(coco_pred, coco_masks, num_classes=50)
        coco_ious.append(coco_miou)
        
        oi_pred = oi_out.argmax(1)
        oi_miou = compute_miou(oi_pred, oi_masks, num_classes=50)
        oi_ious.append(oi_miou)
        
        pbar.set_postfix({'loss': f'{combined_loss.item():.4f}', 
                         'acc': f'{100.*cls_correct/cls_total:.2f}%',
                         'coco_mIoU': f'{coco_miou:.4f}',
                         'oi_mIoU': f'{oi_miou:.4f}'})
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': 100. * cls_correct / cls_total,
        'coco_miou': sum(coco_ious) / len(coco_ious) if coco_ious else 0,
        'oi_miou': sum(oi_ious) / len(oi_ious) if oi_ious else 0
    }

def validate(model, cls_loader, coco_loader, oi_loader, cls_criterion, seg_criterion):
    model.eval()
    total_loss = 0
    cls_correct = 0
    cls_total = 0
    coco_ious = []
    oi_ious = []
    
    with torch.no_grad():
        cls_iter = iter(cls_loader)
        coco_iter = iter(coco_loader)
        oi_iter = iter(oi_loader)
        num_batches = min(len(cls_loader), len(coco_loader), len(oi_loader))
        
        for _ in range(num_batches):
            try:
                cls_imgs, cls_labels = next(cls_iter)
                coco_imgs, coco_masks = next(coco_iter)
                oi_imgs, oi_masks = next(oi_iter)
            except StopIteration:
                break
            
            cls_imgs, cls_labels = cls_imgs.to(device), cls_labels.to(device)
            coco_imgs, coco_masks = coco_imgs.to(device), coco_masks.to(device)
            oi_imgs, oi_masks = oi_imgs.to(device), oi_masks.to(device)
            
            cls_out, _, _ = model(cls_imgs, task='classification')
            _, coco_out, _ = model(coco_imgs, task='segmentation_coco')
            _, _, oi_out = model(oi_imgs, task='segmentation_oi')
            
            cls_loss = cls_criterion(cls_out, cls_labels)
            coco_loss = seg_criterion(coco_out, coco_masks)
            oi_loss = seg_criterion(oi_out, oi_masks)
            
            total_loss += ((cls_loss + coco_loss + oi_loss) / 3.0).item()
            
            _, predicted = cls_out.max(1)
            cls_total += cls_labels.size(0)
            cls_correct += predicted.eq(cls_labels).sum().item()
            
            coco_pred = coco_out.argmax(1)
            coco_miou = compute_miou(coco_pred, coco_masks, num_classes=50)
            coco_ious.append(coco_miou)
            
            oi_pred = oi_out.argmax(1)
            oi_miou = compute_miou(oi_pred, oi_masks, num_classes=50)
            oi_ious.append(oi_miou)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': 100. * cls_correct / cls_total,
        'coco_miou': sum(coco_ious) / len(coco_ious) if coco_ious else 0,
        'oi_miou': sum(oi_ious) / len(oi_ious) if oi_ious else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='muon', choices=['sgd', 'rmsprop', 'adam', 'muon', 'mtmuon'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.02)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"3-Task Learning: ImageNet + COCO + Open Images")
    print(f"{'='*70}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Datasets
    print("Loading datasets...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    imagenet_train = ImageNetDataset('datasets/imageNet', 'train', train_transform, subset_size=50000)
    imagenet_val = ImageNetDataset('datasets/imageNet', 'val', val_transform, subset_size=10000)
    coco_train = COCOSegmentationDataset('datasets/coco 2017', 'train', train_transform, subset_size=50000)
    coco_val = COCOSegmentationDataset('datasets/coco 2017', 'val', val_transform, subset_size=5000)
    oi_train = OpenImagesSegmentationDataset('datasets/open_images', 'train', train_transform, subset_size=50000, num_classes=50)
    oi_val = OpenImagesSegmentationDataset('datasets/open_images', 'val', val_transform, subset_size=5000, num_classes=50)
    
    cls_train_loader = DataLoader(imagenet_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    cls_val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    coco_train_loader = DataLoader(coco_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    coco_val_loader = DataLoader(coco_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    oi_train_loader = DataLoader(oi_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    oi_val_loader = DataLoader(oi_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = MultiTaskMobileNetV3(num_classes=100, num_seg_classes_coco=50, num_seg_classes_oi=50).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'muon':
        optimizer = Muon(model.parameters(), lr=args.lr)
    elif args.optimizer == 'mtmuon':
        optimizer = MTMuon(model.parameters(), lr=args.lr, num_tasks=3)
    
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    print("Starting training...\n")
    results = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        train_metrics = train_epoch(model, cls_train_loader, coco_train_loader, oi_train_loader,
                                    optimizer, cls_criterion, seg_criterion, 
                                    use_mtmuon=(args.optimizer=='mtmuon'))
        val_metrics = validate(model, cls_val_loader, coco_val_loader, oi_val_loader, cls_criterion, seg_criterion)
        
        epoch_time = time.time() - start_time
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, COCO mIoU: {train_metrics['coco_miou']:.4f}, OI mIoU: {train_metrics['oi_miou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, COCO mIoU: {val_metrics['coco_miou']:.4f}, OI mIoU: {val_metrics['oi_miou']:.4f}")
        print(f"Time: {epoch_time:.2f}s\n")
        
        results.append({
            'epoch': epoch+1,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_coco_miou': train_metrics['coco_miou'],
            'val_coco_miou': val_metrics['coco_miou'],
            'train_oi_miou': train_metrics['oi_miou'],
            'val_oi_miou': val_metrics['oi_miou'],
            'time': epoch_time
        })
    
    # Save results
    result_data = {
        'optimizer': args.optimizer,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'final_val_acc': results[-1]['val_acc'],
        'final_val_coco_miou': results[-1]['val_coco_miou'],
        'final_val_oi_miou': results[-1]['val_oi_miou'],
        'avg_epoch_time': sum(r['time'] for r in results) / len(results),
        'history': results
    }
    
    with open(f'results/3task_{args.optimizer}_bs{args.batch_size}.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Val Accuracy: {results[-1]['val_acc']:.2f}%")
    print(f"Final Val COCO mIoU: {results[-1]['val_coco_miou']:.4f}")
    print(f"Final Val OI mIoU: {results[-1]['val_oi_miou']:.4f}")
    print(f"Avg Time/Epoch: {sum(r['time'] for r in results)/len(results):.2f}s")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
