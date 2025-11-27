# Quick Start Guide

## Setup Complete! ✅

Your new ImageNet + COCO2017 multi-task learning system is ready.

## Run Training

### Test Run (2 epochs, fast)
```bash
source venv/bin/activate
python train.py --optimizer adam --batch-size 8 --epochs 2
```

### Full Benchmark (All Optimizers)
```bash
# SGD
python train.py --optimizer sgd --batch-size 8 --epochs 5

# SGD + Nesterov
python train.py --optimizer sgd_nesterov --batch-size 8 --epochs 5

# Adam
python train.py --optimizer adam --batch-size 8 --epochs 5

# AdamW
python train.py --optimizer adamw --batch-size 8 --epochs 5

# RMSprop
python train.py --optimizer rmsprop --batch-size 8 --epochs 5

# Adagrad
python train.py --optimizer adagrad --batch-size 8 --epochs 5

# MTAdamV2 (Equal Weighting)
python train.py --optimizer mtadamv2 --batch-size 8 --epochs 5
```

### Different Batch Sizes
```bash
python train.py --optimizer adam --batch-size 16 --epochs 5
python train.py --optimizer adam --batch-size 32 --epochs 5
```

### Different Models
```bash
# MobileNetV3 (default, faster)
python train.py --optimizer adam --model mobilenet --batch-size 8 --epochs 5

# ResNet-18 (deeper, slower)
python train.py --optimizer adam --model resnet18 --batch-size 8 --epochs 5
```

## What to Expect

### Training Output
- Real-time progress bar with loss, accuracy, mIoU
- Epoch-by-epoch metrics
- Final summary with best results

### Metrics
- **Classification Accuracy**: Top-1 accuracy on ImageNet (100 classes)
- **Segmentation mIoU**: Mean Intersection over Union on COCO (21 classes)
- **Training Time**: Seconds per epoch

### Expected Results (2 epochs, batch 8)
```
Optimizer: adam
Final Validation Accuracy: ~45-50%
Final Validation mIoU: ~0.25-0.35
Average Time/Epoch: ~120-180s
```

## Next Steps

1. Run test with Adam (2 epochs) to verify everything works
2. Run full benchmark with all optimizers (5 epochs each)
3. Compare results and create table
4. Present to faculty!

## Dataset Info

- **ImageNet**: Using 100 classes, 5000 train / 1000 val images
- **COCO2017**: Using 21 classes, 2000 train / 500 val images
- **Image Size**: 256×256 pixels
- **Models**: MobileNetV3 (6.1M params) or ResNet-18 (12.9M params)

Good luck! 🚀
