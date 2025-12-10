# Training Setup Guide

## ✅ Dataset Status: READY

Your datasets are properly integrated:
- **ImageNet**: 1,000 classes with ~1,621 images (sample checked)
- **COCO2017**: 118,287 train images + 5,000 val images with annotations

## 🚀 How to Train

### Method 1: Quick Training (Command Line)

```bash
# Basic training with Adam optimizer
python train.py --optimizer adam --batch-size 8 --epochs 5

# Try different optimizers
python train.py --optimizer sgd --batch-size 8 --epochs 5
python train.py --optimizer adamw --batch-size 8 --epochs 5
python train.py --optimizer mtadamv2 --batch-size 8 --epochs 5

# Different models
python train.py --optimizer adam --model mobilenet --batch-size 8 --epochs 5
python train.py --optimizer adam --model resnet18 --batch-size 8 --epochs 5

# Adjust batch size based on your GPU memory
python train.py --optimizer adam --batch-size 16 --epochs 5  # More GPU memory
python train.py --optimizer adam --batch-size 4 --epochs 5   # Less GPU memory
```

### Method 2: Interactive Training (Recommended for Beginners)

```bash
python train_interactive.py
```

This will guide you through:
1. Model selection (MobileNetV3 or ResNet-18)
2. Batch size selection
3. Optimizer selection (8 options)
4. Number of epochs
5. Automatic learning rate configuration

Results are saved to `results/` folder with plots!

## 📊 Available Optimizers

1. **SGD** - Classic gradient descent
2. **SGD+Nesterov** - SGD with momentum
3. **Adam** - Adaptive learning rate (recommended for beginners)
4. **AdamW** - Adam with weight decay
5. **RMSprop** - Root mean square propagation
6. **Adagrad** - Adaptive gradient
7. **MTAdamV2** - Multi-task Adam (custom optimizer)

## 🎯 Training Configuration

### Dataset Subsets Used
- **ImageNet**: 10,000 train / 2,000 val images (100 classes)
- **COCO2017**: 10,000 train / 1,000 val images (21 classes)
- **Image Size**: 256×256 pixels

### Models Available
- **MobileNetV3-Large**: 6.1M parameters (faster, less memory)
- **ResNet-18**: 12.9M parameters (more accurate, more memory)

### Recommended Settings

**For Quick Testing (2-5 minutes):**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 2
```

**For Full Benchmark (30-60 minutes):**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 10
```

**For GPU with 8GB+ VRAM:**
```bash
python train.py --optimizer adam --batch-size 16 --epochs 10
```

**For GPU with 4GB VRAM:**
```bash
python train.py --optimizer adam --batch-size 4 --epochs 10
```

## 📈 Expected Results

### After 5 Epochs (Batch Size 8)
- **Classification Accuracy**: 45-55%
- **Segmentation mIoU**: 0.25-0.40
- **Time per Epoch**: 2-5 minutes (GPU) / 15-30 minutes (CPU)

### After 10 Epochs (Batch Size 8)
- **Classification Accuracy**: 55-65%
- **Segmentation mIoU**: 0.35-0.50
- **Time per Epoch**: 2-5 minutes (GPU) / 15-30 minutes (CPU)

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train.py --optimizer adam --batch-size 4 --epochs 5
```

### Training Too Slow
```bash
# Use MobileNetV3 instead of ResNet-18
python train.py --optimizer adam --model mobilenet --batch-size 8 --epochs 5
```

### Want Faster Results
```bash
# Reduce epochs for quick testing
python train.py --optimizer adam --batch-size 8 --epochs 2
```

## 📁 Output Files

After training, you'll find:
- **Console Output**: Real-time metrics (loss, accuracy, mIoU)
- **results/*.json**: Detailed training history
- **results/*.png**: Training curves (accuracy & mIoU plots)

## 🎓 Comparing Optimizers

To compare all optimizers, run each one:

```bash
python train.py --optimizer sgd --batch-size 8 --epochs 5
python train.py --optimizer sgd_nesterov --batch-size 8 --epochs 5
python train.py --optimizer adam --batch-size 8 --epochs 5
python train.py --optimizer adamw --batch-size 8 --epochs 5
python train.py --optimizer rmsprop --batch-size 8 --epochs 5
python train.py --optimizer adagrad --batch-size 8 --epochs 5
python train.py --optimizer mtadamv2 --batch-size 8 --epochs 5
```

Then compare the results in the `results/` folder!

## 💡 Tips

1. **Start small**: Test with 2 epochs first to ensure everything works
2. **Monitor GPU usage**: Use `nvidia-smi` (NVIDIA) or Activity Monitor (Mac)
3. **Save results**: Results are automatically saved to `results/` folder
4. **Compare optimizers**: Run multiple optimizers with same settings to compare
5. **Adjust batch size**: Larger = faster but needs more memory

## ✅ Quick Verification

Run this to verify datasets are ready:
```bash
python check_datasets.py
```

You should see:
- [OK] ImageNet found: 1000 classes
- [OK] COCO train images: 118287
- [OK] COCO val images: 5000
- [OK] COCO annotations found

---

**You're all set! Start training now! 🚀**
