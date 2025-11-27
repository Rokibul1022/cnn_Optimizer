"""Multi-Task Models: MobileNetV3 and ResNet-18"""
import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskMobileNetV3(nn.Module):
    """MobileNetV3-Large with classification and segmentation heads"""
    def __init__(self, num_classes=1000, num_seg_classes=21):
        super().__init__()
        
        # Pretrained MobileNetV3 encoder
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.encoder = mobilenet.features  # Shared encoder
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Segmentation head (decoder)
        self.seg_head = nn.Sequential(
            nn.Conv2d(960, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(16, num_seg_classes, 1)
        )
    
    def forward(self, x, task='both'):
        features = self.encoder(x)
        
        if task == 'classification':
            return self.cls_head(features), None
        elif task == 'segmentation':
            return None, self.seg_head(features)
        else:
            return self.cls_head(features), self.seg_head(features)


class MultiTaskResNet18(nn.Module):
    """ResNet-18 with classification and segmentation heads"""
    def __init__(self, num_classes=1000, num_seg_classes=21):
        super().__init__()
        
        # Pretrained ResNet-18 encoder
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Segmentation head (decoder)
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(16, num_seg_classes, 1)
        )
    
    def forward(self, x, task='both'):
        features = self.encoder(x)
        
        if task == 'classification':
            return self.cls_head(features), None
        elif task == 'segmentation':
            return None, self.seg_head(features)
        else:
            return self.cls_head(features), self.seg_head(features)
