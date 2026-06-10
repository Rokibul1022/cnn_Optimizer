# 🚀 Multi-Task Learning: CNN Optimizer Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for benchmarking optimization algorithms on multi-task deep learning. This project simultaneously trains models on **ImageNet classification** and **COCO2017 semantic segmentation** to evaluate optimizer performance in complex, real-world scenarios.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Optimizers](#optimizers)
- [Results](#results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

Multi-task learning presents unique optimization challenges due to competing gradients and varying convergence rates across tasks. This framework provides:

- **Dual-task benchmarking**: Simultaneous ImageNet (classification) + COCO2017 (segmentation)
- **5 optimizer implementations**: SGD, RMSprop, Adam, Muon, MTMuon
- **2 architecture options**: MobileNetV3-Large and ResNet-18
- **Comprehensive metrics**: Accuracy, mIoU, convergence speed, training time
- **Automated visualization**: Training curves, comparison plots, ranking charts

## ✨ Features

- ✅ **Multi-Task Architecture**: Shared encoder with task-specific heads
- ✅ **Flexible Training**: Command-line and interactive modes
- ✅ **GPU Acceleration**: CUDA support with automatic fallback to CPU
- ✅ **Real-time Monitoring**: Progress bars with live metrics
- ✅ **Automatic Checkpointing**: JSON results and training plots
- ✅ **Batch Size Comparison**: Test different batch sizes systematically
- ✅ **Easy Comparison**: Automated multi-optimizer analysis and ranking

## 📁 Project Structure

```
cnn_Optimizer--computer-vision/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── models.py                # Multi-task CNN architectures
│   │                            #   - MultiTaskMobileNetV3
│   │                            #   - MultiTaskResNet18
│   ├── datasets.py              # Dataset loaders
│   │                            #   - ImageNetDataset
│   │                            #   - COCOSegmentationDataset
│   ├── utils.py                 # Utility functions
│   │                            #   - compute_miou()
│   │                            #   - print_results_table()
│   ├── muon.py                  # Muon optimizer implementation
│   ├── mtadam_v2.py            # MTAdamV2 multi-task optimizer
│   └── optimizer_configs.py    # Hyperparameter configurations
├── scripts/                      # Executable scripts
│   ├── train.py                 # Main training pipeline
│   ├── compare_optimizers.py    # Multi-optimizer comparison
│   ├── test_cuda.py             # GPU availability check
│   └── test_setup.py            # Environment verification
├── docs/                         # Documentation
│   └── GUIDE.md                 # Detailed usage guide
├── tests/                        # Unit tests (placeholder)
├── datasets/                     # Dataset directory (not tracked)
│   ├── imageNet/                # ImageNet images
│   └── coco 2017/               # COCO2017 images & annotations
├── results/                      # Training outputs (not tracked)
│   ├── *.json                   # Metrics and history
│   └── *.png                    # Training plots
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installer
├── run.bat                      # Windows quick-launch menu
└── PROJECT_STRUCTURE.md         # Reorganization details
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 16GB+ RAM
- 100GB+ free disk space (for datasets)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd cnn_Optimizer--computer-vision
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python scripts/test_setup.py
python scripts/test_cuda.py  # Check GPU availability
```

## 📦 Dataset Setup

### ImageNet Dataset
1. Download ImageNet dataset (ILSVRC2012)
2. Extract to `datasets/imageNet/`
3. Structure should be:
```
datasets/imageNet/
├── n01440764/  # Class folder
│   ├── image1.JPEG
│   ├── image2.JPEG
│   └── ...
├── n01443537/
└── ...
```

### COCO2017 Dataset
1. Download COCO2017 from [official website](https://cocodataset.org/#download)
   - Train images (118K)
   - Val images (5K)
   - Train/Val annotations
2. Extract to `datasets/coco 2017/`
3. Structure should be:
```
datasets/coco 2017/
├── train2017/
│   └── train2017/
│       ├── 000000000009.jpg
│       └── ...
├── val2017/
│   └── val2017/
│       ├── 000000000139.jpg
│       └── ...
└── annotations_trainval2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

## 🚀 Quick Start

### Option 1: Windows Quick Menu
```bash
run.bat
```
Select from the interactive menu.

### Option 2: Interactive Mode
```bash
python scripts/train.py --interactive
```
Follow the prompts to configure training.

### Option 3: Command Line (Fastest)
```bash
python scripts/train.py --model mobilenet --optimizer adam --batch-size 16 --epochs 5
```

## 💻 Usage

### Training Arguments

```bash
python scripts/train.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | mobilenet | Model architecture (mobilenet/resnet18) |
| `--optimizer` | str | None | Optimizer (sgd/rmsprop/adam/muon/mtmuon) |
| `--batch-size` | int | 16 | Training batch size |
| `--epochs` | int | 5 | Number of training epochs |
| `--lr` | float | 0.02 | Learning rate |
| `--interactive` | flag | False | Use interactive menu |

### Training Examples

**Train MobileNetV3 with Adam:**
```bash
python scripts/train.py --model mobilenet --optimizer adam --batch-size 16 --epochs 10
```

**Train ResNet18 with Muon optimizer:**
```bash
python scripts/train.py --model resnet18 --optimizer muon --batch-size 32 --epochs 15
```

**Train all optimizers on MobileNetV3:**
```bash
for opt in sgd rmsprop adam muon mtmuon; do
    python scripts/train.py --model mobilenet --optimizer $opt --batch-size 16 --epochs 5
done
```

**Custom learning rate:**
```bash
python scripts/train.py --model mobilenet --optimizer sgd --lr 0.001 --epochs 20
```

### Comparing Results

After training multiple configurations:
```bash
python scripts/compare_optimizers.py
```

This generates:
- `results/batch_comparison_mobilenet.png` - Batch size analysis
- `results/batch_comparison_resnet18.png` - Batch size analysis
- `results/model_comparison.png` - Model architecture comparison
- `results/optimizer_ranking.png` - Overall optimizer ranking
- Console table with detailed metrics

## 🏗️ Models

### MobileNetV3-Large
- **Architecture**: Lightweight CNN with inverted residuals
- **Parameters**: ~5.4M (classification) + segmentation head
- **Speed**: Fast inference, suitable for edge devices
- **Use Case**: Resource-constrained environments

### ResNet-18
- **Architecture**: Deep residual network with skip connections
- **Parameters**: ~11.7M (classification) + segmentation head
- **Speed**: Moderate inference time
- **Use Case**: Higher accuracy requirements

**Multi-Task Architecture:**
```
Input Image (256×256)
        ↓
    Encoder (Shared)
    ↙         ↘
Cls Head    Seg Head
    ↓           ↓
Classes    Masks
```

## ⚡ Optimizers

### 1. SGD (Stochastic Gradient Descent)
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Characteristics**: Stable, well-tested baseline
- **Best For**: Large batch sizes, simple objectives

### 2. RMSprop
- **Learning Rate**: 0.0001
- **Characteristics**: Adaptive learning rates per parameter
- **Best For**: Non-stationary objectives, RNNs

### 3. Adam
- **Learning Rate**: 0.0001
- **Characteristics**: Combines momentum and RMSprop
- **Best For**: Sparse gradients, noisy data

### 4. Muon
- **Learning Rate**: 0.02
- **Characteristics**: Momentum-based with orthogonalization
- **Best For**: Deep networks, faster convergence

### 5. MTMuon (Multi-Task Muon)
- **Learning Rate**: 0.02
- **Characteristics**: Nash equilibrium for multi-task learning
- **Best For**: Balancing competing task objectives
- **Special**: Automatic task weight adjustment

## 📊 Results

### Output Files

Each training run generates:

1. **JSON Results** (`results/{model}_{optimizer}_bs{batch}.json`):
```json
{
  "model": "mobilenet",
  "optimizer": "adam",
  "batch_size": 16,
  "final_val_acc": 87.5,
  "final_val_miou": 0.42,
  "best_val_acc": 89.2,
  "best_val_miou": 0.45,
  "convergence_epoch_acc": 7,
  "convergence_epoch_miou": 8,
  "history": [...]
}
```

2. **Training Plots** (`results/{model}_{optimizer}_bs{batch}.png`):
   - Loss curves (train/val)
   - Classification accuracy curves
   - Segmentation mIoU curves
   - Summary statistics table

### Metrics Explained

- **Classification Accuracy**: Top-1 accuracy on ImageNet subset
- **Segmentation mIoU**: Mean Intersection over Union for COCO
- **Convergence Epoch**: Epoch when target performance is reached
  - 85% accuracy for classification
  - 30% mIoU for segmentation
- **Training Time**: Total time per epoch

## 🔬 Advanced Usage

### Custom Dataset Sizes

Edit subset sizes in [train.py](scripts/train.py):
```python
imagenet_train = ImageNetDataset('datasets/imageNet', 'train', 
                                 train_transform, subset_size=50000)  # Change here
coco_train = COCOSegmentationDataset('datasets/coco 2017', 'train', 
                                     train_transform, subset_size=50000)  # Change here
```

### Adding New Optimizers

1. Import optimizer in [train.py](scripts/train.py)
2. Add case in train_single_optimizer():
```python
elif optimizer_name == 'your_optimizer':
    optimizer = YourOptimizer(model.parameters(), lr=0.001)
```

### Modifying Task Weights

In [train_epoch()](scripts/train.py):
```python
combined_loss = cls_loss + 3.0 * seg_loss  # Adjust weight here
```

### Multi-GPU Training

Wrap model with DataParallel:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 8`
- Use gradient accumulation
- Try MobileNetV3 instead of ResNet18

### Dataset Not Found
- Verify dataset paths in code match your directory structure
- Check folder names match exactly (case-sensitive on Linux)

### Slow Training
- Verify CUDA is available: `python scripts/test_cuda.py`
- Reduce `num_workers` in DataLoader if CPU bottleneck
- Use smaller subset sizes for quick tests

### Import Errors
```bash
# Make sure you're in the project root
cd cnn_Optimizer--computer-vision

# Run from root directory
python scripts/train.py
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for the framework
- COCO dataset maintainers
- ImageNet dataset creators
- Optimizer paper authors

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Training! 🎉**
