"""Simple Dataset Check"""
import os

print("\n" + "="*70)
print("DATASET VERIFICATION")
print("="*70)

# Check ImageNet
imagenet_path = 'datasets/imageNet'
if os.path.exists(imagenet_path):
    classes = [d for d in os.listdir(imagenet_path) if os.path.isdir(os.path.join(imagenet_path, d)) and not d.startswith('.')]
    print(f"\n[OK] ImageNet found: {len(classes)} classes")
    
    # Count images in first 3 classes
    total_images = 0
    for cls in classes[:3]:
        cls_path = os.path.join(imagenet_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.JPEG', '.jpg', '.png'))]
        total_images += len(images)
    print(f"[OK] Sample check: {total_images} images in first 3 classes")
else:
    print(f"\n[ERROR] ImageNet NOT found at: {imagenet_path}")

# Check COCO
coco_path = 'datasets/coco 2017'
if os.path.exists(coco_path):
    train_imgs = os.path.join(coco_path, 'train2017', 'train2017')
    val_imgs = os.path.join(coco_path, 'val2017', 'val2017')
    annotations = os.path.join(coco_path, 'annotations_trainval2017', 'annotations')
    
    if os.path.exists(train_imgs):
        train_count = len([f for f in os.listdir(train_imgs) if f.endswith('.jpg')])
        print(f"\n[OK] COCO train images: {train_count}")
    else:
        print(f"\n[ERROR] COCO train images NOT found")
    
    if os.path.exists(val_imgs):
        val_count = len([f for f in os.listdir(val_imgs) if f.endswith('.jpg')])
        print(f"[OK] COCO val images: {val_count}")
    else:
        print(f"[ERROR] COCO val images NOT found")
    
    if os.path.exists(annotations):
        train_ann = os.path.join(annotations, 'instances_train2017.json')
        val_ann = os.path.join(annotations, 'instances_val2017.json')
        if os.path.exists(train_ann) and os.path.exists(val_ann):
            print(f"[OK] COCO annotations found")
        else:
            print(f"[ERROR] COCO annotations NOT found")
    else:
        print(f"[ERROR] COCO annotations directory NOT found")
else:
    print(f"\n[ERROR] COCO NOT found at: {coco_path}")

print("\n" + "="*70)
print("READY TO TRAIN!")
print("="*70)
print("\nRun one of these commands:")
print("  python train.py --optimizer adam --batch-size 8 --epochs 5")
print("  python train_interactive.py")
print("="*70 + "\n")
