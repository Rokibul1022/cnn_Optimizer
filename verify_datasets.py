"""Dataset Verification Script"""
import os
from datasets import ImageNetDataset, COCOSegmentationDataset, get_transforms
from torch.utils.data import DataLoader
import torch

def check_directory(path, name):
    """Check if directory exists and is accessible"""
    if os.path.exists(path):
        print(f"✓ {name} directory found: {path}")
        return True
    else:
        print(f"✗ {name} directory NOT found: {path}")
        return False

def verify_imagenet():
    """Verify ImageNet dataset"""
    print("\n" + "="*70)
    print("VERIFYING IMAGENET DATASET")
    print("="*70)
    
    base_path = 'datasets/imageNet'
    
    # Check directories
    if not check_directory(base_path, "ImageNet base"):
        return False
    
    # Count classes (flat structure)
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]
    
    print(f"✓ Classes found: {len(classes)}")
    
    # Try loading dataset
    try:
        print("\n🔄 Loading ImageNet dataset (500 classes, 50000 images)...")
        transform = get_transforms('train')
        dataset = ImageNetDataset(base_path, 'train', transform, subset_size=50000)
        
        # Test dataloader
        print("🔄 Testing DataLoader...")
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Image range: [{images.min():.3f}, {images.max():.3f}]")
        
        print("\n✅ ImageNet dataset verification PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ImageNet dataset verification FAILED: {str(e)}")
        return False

def verify_coco():
    """Verify COCO2017 dataset"""
    print("\n" + "="*70)
    print("VERIFYING COCO2017 DATASET")
    print("="*70)
    
    base_path = 'datasets/coco 2017'
    train_img_path = os.path.join(base_path, 'train2017', 'train2017')
    val_img_path = os.path.join(base_path, 'val2017', 'val2017')
    ann_path = os.path.join(base_path, 'annotations_trainval2017', 'annotations')
    
    # Check directories
    if not check_directory(base_path, "COCO base"):
        return False
    if not check_directory(train_img_path, "COCO train images"):
        return False
    if not check_directory(val_img_path, "COCO val images"):
        return False
    if not check_directory(ann_path, "COCO annotations"):
        return False
    
    # Check annotation files
    train_ann = os.path.join(ann_path, 'instances_train2017.json')
    val_ann = os.path.join(ann_path, 'instances_val2017.json')
    
    if not os.path.exists(train_ann):
        print(f"✗ Train annotations NOT found: {train_ann}")
        return False
    print(f"✓ Train annotations found: {train_ann}")
    
    if not os.path.exists(val_ann):
        print(f"✗ Val annotations NOT found: {val_ann}")
        return False
    print(f"✓ Val annotations found: {val_ann}")
    
    # Count images
    train_images = len([f for f in os.listdir(train_img_path) if f.endswith('.jpg')])
    val_images = len([f for f in os.listdir(val_img_path) if f.endswith('.jpg')])
    
    print(f"✓ Train images found: {train_images}")
    print(f"✓ Val images found: {val_images}")
    
    # Try loading dataset
    try:
        print("\n🔄 Loading COCO train dataset (50000 images)...")
        transform = get_transforms('train')
        train_dataset = COCOSegmentationDataset(base_path, 'train', transform, subset_size=50000)
        
        print("🔄 Loading COCO val dataset (5000 images)...")
        val_dataset = COCOSegmentationDataset(base_path, 'val', get_transforms('val'), subset_size=5000)
        
        # Test dataloader
        print("🔄 Testing DataLoader...")
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Get one batch
        images, masks = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Masks shape: {masks.shape}")
        print(f"✓ Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"✓ Mask classes: {masks.unique().tolist()}")
        
        print("\n✅ COCO2017 dataset verification PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ COCO2017 dataset verification FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("🔍 DATASET VERIFICATION TOOL")
    print("="*70)
    
    imagenet_ok = verify_imagenet()
    coco_ok = verify_coco()
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"ImageNet: {'✅ PASSED' if imagenet_ok else '❌ FAILED'}")
    print(f"COCO2017: {'✅ PASSED' if coco_ok else '❌ FAILED'}")
    
    if imagenet_ok and coco_ok:
        print("\n✅ All datasets verified successfully!")
        print("You can now run: python train_interactive.py")
    else:
        print("\n❌ Some datasets failed verification.")
        print("Please check the dataset paths and structure.")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
