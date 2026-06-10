"""Test CUDA Setup"""
import torch

print("\n" + "="*70)
print("CUDA CONFIGURATION TEST")
print("="*70)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test tensor on GPU
    print("\nTesting GPU tensor operations...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"[OK] GPU tensor test passed!")
    print(f"[OK] Tensor device: {z.device}")
else:
    print("\n[WARNING] CUDA not available. Training will use CPU.")
    print("Make sure you installed PyTorch with CUDA support:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*70)
print("Device that will be used for training:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> {device}")
print("="*70 + "\n")
