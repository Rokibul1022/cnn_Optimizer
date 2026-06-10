"""Test if setup is working"""
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test imports
try:
    from models import MultiTaskMobileNetV3, MultiTaskResNet18
    from datasets import ImageNetDataset, COCOSegmentationDataset
    from mtadam_v2 import MTAdamV2
    print("\n✓ All imports successful!")
    
    # Test model creation
    model = MultiTaskMobileNetV3(num_classes=100, num_seg_classes=21)
    print(f"✓ MobileNetV3 created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model = MultiTaskResNet18(num_classes=100, num_seg_classes=21)
    print(f"✓ ResNet-18 created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test optimizer
    optimizer = MTAdamV2(model.parameters(), lr=0.001, num_tasks=2)
    print(f"✓ MTAdamV2 created with weights: {optimizer.get_task_weights()}")
    
    print("\n✅ Setup complete! Ready to train.")
    print("\nRun: python train.py --optimizer adam --batch-size 8 --epochs 2")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
