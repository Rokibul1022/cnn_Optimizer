"""Dataset loaders for ImageNet and COCO2017"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np

class ImageNetDataset(Dataset):
    """ImageNet classification dataset"""
    def __init__(self, root_dir, split='train', transform=None, subset_size=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class folders (ImageNet has flat structure with class folders)
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')])[:1000]
        
        images_per_class = max(1, subset_size // len(classes))
        
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            class_count = 0
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.JPEG', '.jpg', '.png')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(idx)
                    class_count += 1
                    if class_count >= images_per_class:
                        break
        
        print(f"Loaded {len(self.images)} ImageNet images, {len(set(self.labels))} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


class COCOSegmentationDataset(Dataset):
    """COCO2017 segmentation dataset"""
    def __init__(self, root_dir, split='train', transform=None, subset_size=5000):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # COCO paths
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train2017', 'train2017')
            ann_file = os.path.join(root_dir, 'annotations_trainval2017', 'annotations', 'instances_train2017.json')
        else:
            self.img_dir = os.path.join(root_dir, 'val2017', 'val2017')
            ann_file = os.path.join(root_dir, 'annotations_trainval2017', 'annotations', 'instances_val2017.json')
        
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())[:subset_size]
        
        print(f"Loaded {len(self.img_ids)} COCO images")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                mask = np.maximum(mask, self.coco.annToMask(ann) * ann['category_id'])
        
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


def get_transforms(split='train'):
    """Get data transforms"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
