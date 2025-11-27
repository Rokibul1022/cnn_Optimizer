# Multi-Task Learning: ImageNet + COCO2017

Optimizer Performance Benchmarking Framework for Multi-Task Deep Learning

## Project Overview

- **Tasks**: Image Classification (ImageNet 1000 classes) + Semantic Segmentation (COCO 21 classes)
- **Models**: MobileNetV3-Large, ResNet-18
- **Optimizers**: SGD, SGD+Nesterov, Adam, AdamW, RMSprop, Adagrad, Adadelta, MTAdamV2
- **Framework**: PyTorch with CUDA support

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/optimizer_cnn.git
cd optimizer_cnn
```

### 2. Setup Environment

**Windows (CUDA):**
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Mac (MPS):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Datasets

Download and place datasets in `datasets/` folder:

```
datasets/
├── imageNet/
│   ├── n01440764/  (class folders)
│   ├── n01443537/
│   └── ...
└── coco 2017/
    ├── train2017/
    ├── val2017/
    └── annotations_trainval2017/
```

### 4. Verify Setup

```bash
python check_gpu.py
python verify_datasets.py
```

### 5. Train

```bash
python train_interactive.py
```

### 6. Compare Results

```bash
python comparison.py
```

## Files

- `train_interactive.py` - Interactive training script
- `comparison.py` - Results comparison and visualization
- `models.py` - MobileNetV3 and ResNet-18 architectures
- `datasets.py` - ImageNet and COCO data loaders
- `mtadam_v2.py` - MTAdamV2 optimizer (equal weighting)
- `utils.py` - Helper functions (mIoU calculation)
- `verify_datasets.py` - Dataset verification
- `check_gpu.py` - GPU detection

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 12GB+ GPU memory (recommended)
- ImageNet dataset
- COCO2017 dataset

## Expected Performance

### RTX 3060 (12GB VRAM)
- Batch Size 32: ~60s/epoch
- Full benchmark (48 experiments): 3-4 hours

### M1 MacBook Air
- Batch Size 8: ~150s/epoch
- Full benchmark (48 experiments): 18-24 hours

## Citation

If you use this code, please cite:
```
@misc{optimizer_cnn_2024,
  title={Multi-Task Learning Optimizer Comparison},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
