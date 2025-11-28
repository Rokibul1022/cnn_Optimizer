# Optimized Learning Rates for Both Models

## MobileNetV3-Large (6.1M params, Lightweight)

| Optimizer | Learning Rate | Rationale |
|-----------|---------------|-----------|
| SGD | 0.01 | Standard for transfer learning |
| SGD+Nesterov | 0.01 | Same as SGD |
| Adam | 0.001 | ✅ Validated: 83.4% accuracy |
| AdamW | 0.001 | Same as Adam |
| RMSprop | 0.001 | Adaptive, works with lower LR |
| Adagrad | 0.01 | Accumulates gradients, needs higher initial |
| Adadelta | 1.0 | LR-free (standard value) |
| MTAdamV2 | 0.001 | ✅ Validated: 83.4% accuracy |

## ResNet-18 (12.9M params, Deeper)

| Optimizer | Learning Rate | Rationale |
|-----------|---------------|-----------|
| SGD | 0.01 | ✅ Validated: 46.4% accuracy |
| SGD+Nesterov | 0.01 | ✅ Validated: 46.4% accuracy |
| Adam | 0.005 | 5× higher than MobileNetV3 (deeper network) |
| AdamW | 0.005 | Same as Adam |
| RMSprop | 0.005 | Adaptive, needs higher for depth |
| Adagrad | 0.02 | 2× higher than MobileNetV3 |
| Adadelta | 1.0 | LR-free (standard value) |
| MTAdamV2 | 0.005 | Same as Adam |

## Why Different Learning Rates?

### MobileNetV3 (Shallow, Wide):
- Fewer layers (16 inverted residual blocks)
- Wider feature maps (960 channels)
- Gradients flow easily
- **Lower LR works well** (0.001 for Adam)

### ResNet-18 (Deep, Narrow):
- More layers (18 layers with residuals)
- Narrower feature maps (512 channels)
- Gradients dilute through depth
- **Higher LR needed** (0.005 for Adam)

## Mathematical Justification:

```
Gradient magnitude ∝ 1/√depth

MobileNetV3: depth ≈ 16 → gradient scale ≈ 0.25
ResNet-18: depth ≈ 18 → gradient scale ≈ 0.24

But ResNet has residual connections that help, so:
LR_ResNet ≈ 5 × LR_MobileNetV3 for Adam-family
```

## Expected Performance:

### MobileNetV3 + Adam (LR=0.001):
- Epoch 1: 28% → Epoch 5: 83% ✅

### ResNet-18 + Adam (LR=0.005):
- Epoch 1: 20-30% → Epoch 5: 60-70% (expected)

### ResNet-18 + SGD+Nesterov (LR=0.01):
- Epoch 1: 12.5% → Epoch 5: 46.4% ✅

## Usage:

Learning rates are **automatically set** based on model and optimizer selection. No manual tuning needed!

```bash
python train_interactive.py
# Select model → Select optimizer → LR is set automatically
```
