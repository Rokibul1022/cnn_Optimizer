# 🚀 START HERE - Quick Training Guide

## ✅ Your Project is Ready!

Datasets verified and integrated:
- ✅ ImageNet: 1,000 classes
- ✅ COCO2017: 118,287 train + 5,000 val images

## 🎯 Start Training in 3 Steps

### Step 1: Open Terminal/Command Prompt
Navigate to project folder:
```bash
cd c:\Users\User\Desktop\optimizer_cnn\cnn_Optimizer
```

### Step 2: Choose Your Training Method

**Option A - Quick Test (Recommended First):**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 2
```
⏱️ Takes ~5-10 minutes

**Option B - Interactive Mode:**
```bash
python train_interactive.py
```
📋 Guides you through all options

**Option C - Full Training:**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 10
```
⏱️ Takes ~30-60 minutes

### Step 3: View Results
Results saved in `results/` folder:
- JSON files with metrics
- PNG plots showing training curves

## 📊 What You'll See

During training:
```
Epoch 1/5
Training: 100%|████████| loss: 2.3456, acc: 45.23%, mIoU: 0.3214
Train - Loss: 2.3456, Acc: 45.23%, mIoU: 0.3214
Val   - Loss: 2.4567, Acc: 43.12%, mIoU: 0.3012
Time: 180.45s
```

## 🎓 Compare All Optimizers

Run each optimizer to compare performance:
```bash
python train.py --optimizer sgd --batch-size 8 --epochs 5
python train.py --optimizer adam --batch-size 8 --epochs 5
python train.py --optimizer adamw --batch-size 8 --epochs 5
python train.py --optimizer mtadamv2 --batch-size 8 --epochs 5
```

## 🔧 Common Commands

**Test if everything works:**
```bash
python check_datasets.py
```

**Quick 2-epoch test:**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 2
```

**Full benchmark:**
```bash
python train.py --optimizer adam --batch-size 8 --epochs 10
```

**Use different model:**
```bash
python train.py --optimizer adam --model resnet18 --batch-size 8 --epochs 5
```

**Adjust for GPU memory:**
```bash
# More memory available
python train.py --optimizer adam --batch-size 16 --epochs 5

# Less memory available
python train.py --optimizer adam --batch-size 4 --epochs 5
```

## 📖 Need More Details?

Read `TRAINING_GUIDE.md` for:
- Detailed optimizer explanations
- Expected results
- Troubleshooting tips
- Advanced configurations

## ⚡ Quick Start Now!

```bash
python train.py --optimizer adam --batch-size 8 --epochs 2
```

This will train for 2 epochs (~5-10 minutes) to verify everything works!

---

**Ready? Let's train! 🎉**
