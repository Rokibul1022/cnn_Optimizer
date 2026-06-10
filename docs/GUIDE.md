# CNN Optimizer Project - Complete Guide

## Overview
This project benchmarks different optimizers on multi-task learning (classification + segmentation).

## Quick Start

### Option 1: Interactive Mode
```bash
python scripts/train.py --interactive
```

### Option 2: Command Line
```bash
python scripts/train.py --model mobilenet --optimizer adam --batch-size 16 --epochs 5
```

### Option 3: Windows Batch Script
```bash
run.bat
```

## Available Options

### Models
- `mobilenet` - MobileNetV3-Large (faster, fewer parameters)
- `resnet18` - ResNet-18 (deeper, more accurate)

### Optimizers
- `sgd` - Stochastic Gradient Descent
- `rmsprop` - RMSprop
- `adam` - Adam
- `muon` - Muon optimizer
- `mtmuon` - Multi-Task Muon

### Hyperparameters
- `--batch-size` - Batch size (default: 16)
- `--epochs` - Number of epochs (default: 5)
- `--lr` - Learning rate (default: 0.02)

## Dataset Setup

Place datasets in the following structure:
```
datasets/
├── imageNet/
│   ├── class1/
│   ├── class2/
│   └── ...
└── coco 2017/
    ├── train2017/
    ├── val2017/
    └── annotations_trainval2017/
```

## Results

Results are saved to `results/` folder:
- JSON files with metrics
- PNG plots with training curves

## Comparison

After training multiple optimizers:
```bash
python scripts/compare_optimizers.py
```

This generates:
- Performance comparison tables
- Batch size comparison plots
- Model comparison plots
- Optimizer ranking
