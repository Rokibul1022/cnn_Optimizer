# 🚀 Complete Training System - Instructions

## System Overview

This system provides:
- **8 Optimizers**: SGD, SGD+Nesterov, Adam, AdamW, RMSprop, Adagrad, Adadelta, MTAdamV2
- **2 Models**: MobileNetV3-Large (6.1M params), ResNet-18 (12.9M params)
- **Interactive Training**: Choose model, optimizer, batch size, epochs
- **Automatic Results Tracking**: JSON files, plots, and charts saved per experiment
- **Comprehensive Comparison**: Analyze all experiments with tables and visualizations

## Files Created

1. **train_interactive.py** - Interactive training with user prompts
2. **comparison.py** - Compare all results with graphs and tables
3. **verify_datasets.py** - Verify datasets are properly loaded
4. **results/** - Folder where all results are saved

## Step 1: Verify Datasets ✓

```bash
python3 verify_datasets.py
```

This checks:
- ImageNet directory structure
- COCO2017 directory structure
- Ability to load batches from both datasets

## Step 2: Run Interactive Training

```bash
python3 train_interactive.py
```

You'll be prompted to select:
1. **Model**: MobileNetV3 (1) or ResNet-18 (2)
2. **Batch Size**: 4, 8, 16, or 32
3. **Optimizer**: Choose from 8 options
4. **Epochs**: Number of training epochs
5. **Learning Rate**: Default 0.001

### Example Session:
```
📦 SELECT MODEL:
1. MobileNetV3-Large (6.1M params, faster)
2. ResNet-18 (12.9M params, more accurate)
Enter choice (1 or 2): 1

📊 SELECT BATCH SIZE:
Recommended: 4, 8, 16, 32
Enter batch size: 8

⚙️  SELECT OPTIMIZER:
1. SGD
2. SGD+Nesterov
3. Adam
4. AdamW
5. RMSprop
6. Adagrad
7. Adadelta
8. MTAdamV2
Enter choice (1-8): 3

🔄 SELECT EPOCHS:
Enter number of epochs: 5

📈 SELECT LEARNING RATE:
Enter learning rate (default 0.001): 0.001
```

### Results Saved:
- `results/mobilenetv3_adam_bs8.json` - Training metrics
- `results/mobilenetv3_adam_bs8.png` - Accuracy and mIoU plots

## Step 3: Run Multiple Experiments

Train with different configurations:

```bash
# Example 1: MobileNetV3 + Adam + BS8
python3 train_interactive.py
# Select: 1, 8, 3, 5, 0.001

# Example 2: ResNet-18 + AdamW + BS16
python3 train_interactive.py
# Select: 2, 16, 4, 5, 0.001

# Example 3: MobileNetV3 + MTAdamV2 + BS8
python3 train_interactive.py
# Select: 1, 8, 8, 5, 0.001
```

## Step 4: Compare All Results

After running multiple experiments:

```bash
python3 comparison.py
```

This generates:
- **Summary Table**: All experiments with metrics
- **optimizer_comparison.png**: Compare optimizers (accuracy, mIoU, time, efficiency)
- **batch_size_comparison.png**: Performance vs batch size
- **model_comparison.png**: MobileNetV3 vs ResNet-18
- **convergence_curves.png**: Training curves for all experiments
- **summary_report.txt**: Text report with best configurations

## Recommended Benchmark Suite

### Quick Test (2-3 hours):
```
Model: MobileNetV3
Batch Size: 8
Epochs: 5
Optimizers: All 8 (run train_interactive.py 8 times)
```

### Full Benchmark (6-9 hours):
```
Models: Both (MobileNetV3 + ResNet-18)
Batch Sizes: 8, 16
Epochs: 5
Optimizers: All 8
Total: 2 models × 2 batch sizes × 8 optimizers = 32 experiments
```

## Expected Results

### MobileNetV3 (Batch Size 8, 5 Epochs):
- Classification Accuracy: 45-50%
- Segmentation mIoU: 0.25-0.35
- Time per Epoch: 120-150s

### ResNet-18 (Batch Size 8, 5 Epochs):
- Classification Accuracy: 48-53%
- Segmentation mIoU: 0.28-0.38
- Time per Epoch: 150-180s

## Troubleshooting

### Dataset Issues:
```bash
# Check dataset paths
ls -la datasets/imageNet/
ls -la "datasets/coco 2017/"

# Run verification
python3 verify_datasets.py
```

### Memory Issues:
- Reduce batch size to 4
- Use MobileNetV3 instead of ResNet-18

### Slow Training:
- Reduce subset_size in datasets.py
- Use fewer epochs for testing

## Results Structure

```
results/
├── mobilenetv3_adam_bs8.json          # Training metrics
├── mobilenetv3_adam_bs8.png           # Individual plots
├── resnet18_adamw_bs16.json
├── resnet18_adamw_bs16.png
├── optimizer_comparison.png            # Comparison plots
├── batch_size_comparison.png
├── model_comparison.png
├── convergence_curves.png
└── summary_report.txt                  # Text summary
```

## Quick Commands

```bash
# Verify datasets
python3 verify_datasets.py

# Train (interactive)
python3 train_interactive.py

# Compare results
python3 comparison.py

# View results
ls -lh results/
cat results/summary_report.txt
```

## Notes

- All results are automatically saved to `results/` folder
- Each experiment creates a JSON file and PNG plot
- comparison.py can be run anytime to analyze existing results
- Dataset verification should pass before training
- Use batch size 8 for balanced speed/memory usage
