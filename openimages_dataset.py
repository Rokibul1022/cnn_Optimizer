"""Open Images Dataset Loader for Segmentation"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class OpenImagesSegmentationDataset(Dataset):
    """
    Open Images segmentation dataset
    Falls back to using COCO dataset if Open Images not available
    """
    def __init__(self, root_dir, split='train', transform=None, subset_size=50000, num_classes=50):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # Check if Open Images exists
        self.img_dir = os.path.join(root_dir, split, 'images')
        
        if os.path.exists(self.img_dir):
            all_images = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.JPEG'))]
            self.images = []
            while len(self.images) < subset_size:
                self.images.extend(all_images)
            self.images = self.images[:subset_size]
            self.use_coco_fallback = False
            print(f"Loaded {len(self.images)} Open Images images ({len(all_images)} unique)")
        else:
            # Fallback: use COCO dataset as second segmentation dataset
            print(f"WARNING: Open Images not found at {self.img_dir}")
            print(f"Using COCO dataset as fallback for second segmentation task")
            from pycocotools.coco import COCO
            
            coco_root = 'datasets/coco 2017'
            if split == 'train':
                coco_img_dir = os.path.join(coco_root, 'train2017', 'train2017')
                ann_file = os.path.join(coco_root, 'annotations_trainval2017', 'annotations', 'instances_train2017.json')
            else:
                coco_img_dir = os.path.join(coco_root, 'val2017', 'val2017')
                ann_file = os.path.join(coco_root, 'annotations_trainval2017', 'annotations', 'instances_val2017.json')
            
            self.coco = COCO(ann_file)
            self.coco_img_dir = coco_img_dir
            all_img_ids = list(self.coco.imgs.keys())
            
            # Repeat to reach subset_size
            self.img_ids = []
            while len(self.img_ids) < subset_size:
                self.img_ids.extend(all_img_ids)
            self.img_ids = self.img_ids[:subset_size]
            self.use_coco_fallback = True
            print(f"Loaded {len(self.img_ids)} COCO images as Open Images fallback ({len(all_img_ids)} unique)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.use_coco_fallback:
            # Use COCO data
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.coco_img_dir, img_info['file_name'])
            
            image = Image.open(img_path).convert('RGB')
            
            # Load annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Create segmentation mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in anns:
                if 'segmentation' in ann:
                    cat_id = ann['category_id']
                    cat_id = min(cat_id, self.num_classes - 1)
                    mask = np.maximum(mask, self.coco.annToMask(ann) * cat_id)
            
            mask = Image.fromarray(mask)
            
            if self.transform:
                image = self.transform(image)
                mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
                mask = torch.from_numpy(np.array(mask)).long()
                mask = torch.clamp(mask, 0, self.num_classes - 1)
            
            return image, mask
        else:
            # Use Open Images data
            img_name = self.images[idx]
            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.root_dir, self.split, 'masks', img_name.replace('.jpg', '.png'))
            
            image = Image.open(img_path).convert('RGB')
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask)
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            
            mask = np.clip(mask, 0, self.num_classes - 1)
            mask = Image.fromarray(mask)
            
            if self.transform:
                image = self.transform(image)
                mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
                mask = torch.from_numpy(np.array(mask)).long()
                mask = torch.clamp(mask, 0, self.num_classes - 1)
            
            return image, mask
