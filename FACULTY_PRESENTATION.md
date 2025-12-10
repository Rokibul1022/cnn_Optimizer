# Multi-Task Learning: Optimizer Comparison Study

## Project Overview
**Objective**: Compare optimizer performance on multi-task deep learning (classification + segmentation)

## Dataset Configuration
- **ImageNet Classification**: 100 classes, 50,000 train / 10,000 validation images
- **COCO2017 Segmentation**: 21 classes, 50,000 train / 10,000 validation images
- **Image Resolution**: 256×256 pixels
- **Total Training Samples**: 100,000 images (50k per task)

## Model Architecture
- **Base Model**: MobileNetV3-Large (pretrained on ImageNet)
- **Parameters**: ~6.1M trainable parameters
- **Tasks**: 
  - Classification head: 100-class softmax
  - Segmentation head: 21-class pixel-wise prediction
- **Multi-task Learning**: Shared encoder, separate task-specific heads

## Training Configuration
- **Batch Size**: 32
- **Epochs**: 10
- **Hardware**: NVIDIA RTX 3060 (12GB VRAM)
- **Training Time**: ~40 minutes per optimizer
- **Total Experiments**: 4 optimizers

## Optimizers Compared

### 1. SGD (Stochastic Gradient Descent)
- Learning Rate: 0.01
- Momentum: 0.9
- Classic baseline optimizer

### 2. Adam (Adaptive Moment Estimation)
- Learning Rate: 0.001
- β1=0.9, β2=0.999
- Industry standard adaptive optimizer

### 3. AdamW (Adam with Weight Decay)
- Learning Rate: 0.001
- Weight Decay: 0.01
- Improved regularization over Adam

### 4. MTAdamV2 (Multi-Task Adam V2)
- Learning Rate: 0.0003
- Custom optimizer for multi-task learning
- Automatic task balancing

## Evaluation Metrics

### Classification Task
- **Top-1 Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss

### Segmentation Task
- **Mean IoU (mIoU)**: Intersection over Union averaged across classes
- **Loss**: Pixel-wise cross-entropy

### Overall Performance
- **Combined Loss**: Weighted sum of both tasks
- **Training Time**: Seconds per epoch
- **Convergence Speed**: Epochs to reach target performance

## Expected Results

### Classification Accuracy (10 epochs)
- SGD: 35-45%
- Adam: 45-55%
- AdamW: 45-55%
- MTAdamV2: 40-50%

### Segmentation mIoU (10 epochs)
- SGD: 0.15-0.25
- Adam: 0.25-0.35
- AdamW: 0.25-0.35
- MTAdamV2: 0.30-0.40

## Key Findings (To be filled after training)

1. **Best Overall Performance**: [Optimizer name]
2. **Fastest Convergence**: [Optimizer name]
3. **Best Classification**: [Optimizer name]
4. **Best Segmentation**: [Optimizer name]
5. **Most Stable Training**: [Optimizer name]

## Technical Contributions

1. **Multi-Task Framework**: Unified training pipeline for classification + segmentation
2. **Optimizer Comparison**: Fair comparison with identical hyperparameters
3. **Scalable Architecture**: MobileNetV3 for efficient training
4. **Reproducible Results**: Fixed seeds, documented configuration

## Challenges & Solutions

### Challenge 1: Dataset Imbalance
- **Problem**: COCO has 118k images, ImageNet subset has 50k
- **Solution**: Repeated COCO samples to match ImageNet size

### Challenge 2: Task Balancing
- **Problem**: Classification and segmentation have different loss scales
- **Solution**: MTAdamV2 optimizer with automatic task weighting

### Challenge 3: Training Time
- **Problem**: Limited time for experiments
- **Solution**: Reduced dataset to 50k samples, batch size 32, GPU acceleration

## Future Work

1. Increase dataset size to full ImageNet (1000 classes)
2. Test on larger models (ResNet-50, EfficientNet)
3. Implement advanced multi-task learning techniques
4. Extend to more tasks (depth estimation, object detection)

## Conclusion

This study demonstrates the effectiveness of different optimizers in multi-task learning scenarios. Results show [to be filled] performs best for simultaneous classification and segmentation tasks.

---

**Training Time**: ~2.7 hours (4 optimizers × 40 min)
**Total Experiments**: 4
**Hardware**: NVIDIA RTX 3060
**Framework**: PyTorch 2.7.1
