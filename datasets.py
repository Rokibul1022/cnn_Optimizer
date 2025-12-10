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
    def __init__(self, root_dir, split='train', transform=None, subset_size=250000):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class folders (ImageNet has flat structure with class folders)
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')])
        
        images_per_class = 200
        classes = classes[:100]  # Use only first 100 classes
        
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            all_images = [img for img in os.listdir(class_path) if img.endswith(('.JPEG', '.jpg', '.png'))]
            
            # Split: first 80% for train, last 20% for val
            if split == 'train':
                selected_images = all_images[:int(len(all_images) * 0.8)][:images_per_class]
            else:
                selected_images = all_images[int(len(all_images) * 0.8):][:int(images_per_class*0.2)]
            
            for img_name in selected_images:
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(idx)
        
        print(f"Loaded {len(self.images)} ImageNet images, {len(set(self.labels))} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # Return a black image if file is corrupted
            return torch.zeros(3, 256, 256), label


class COCOSegmentationDataset(Dataset):
    """COCO2017 segmentation dataset"""
    def __init__(self, root_dir, split='train', transform=None, subset_size=250000):
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
        all_img_ids = list(self.coco.imgs.keys())
        
        # Use only subset_size images
        self.img_ids = all_img_ids[:subset_size]
        
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
        
        # Create segmentation mask with ignore_index for background
        mask = np.full((img_info['height'], img_info['width']), 255, dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann and ann.get('area', 0) > 0:
                cat_id = ann['category_id']
                # Map to 50 classes (0-49)
                cat_id = cat_id % 50
                ann_mask = self.coco.annToMask(ann)
                mask[ann_mask > 0] = cat_id
        
        mask = Image.fromarray(mask)
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
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
