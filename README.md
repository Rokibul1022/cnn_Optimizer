# Multi-Task Learning: ImageNet + COCO2017

## Optimizer Performance Benchmarking Framework

### Datasets
- **ImageNet**: 1000-class image classification
- **COCO2017**: 21-class semantic segmentation

### Models
- MobileNetV3-Large (lightweight)
- ResNet-18 (deeper)

### Optimizers Tested
1. SGD
2. SGD+Nesterov
3. Adam
4. AdamW
5. RMSprop
6. Adagrad
7. MTAdamV2 (Equal Weighting)

### Metrics
- Classification: Top-1 Accuracy
- Segmentation: Mean IoU
- Convergence Speed
- Training Time

### Quick Start
```bash
python train.py --optimizer adam --batch-size 8 --epochs 10
```
