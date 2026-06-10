@echo off
echo CNN Optimizer - Multi-Task Learning
echo ====================================
echo.
echo 1. Train MobileNetV3 with Adam
echo 2. Train ResNet18 with Adam
echo 3. Train with custom settings
echo 4. Compare all results
echo 5. Test CUDA
echo 0. Exit
echo.
set /p choice=Enter choice: 

if "%choice%"=="1" (
    python scripts\train.py --model mobilenet --optimizer adam --batch-size 16 --epochs 5
) else if "%choice%"=="2" (
    python scripts\train.py --model resnet18 --optimizer adam --batch-size 16 --epochs 5
) else if "%choice%"=="3" (
    python scripts\train.py --interactive
) else if "%choice%"=="4" (
    python scripts\compare_optimizers.py
) else if "%choice%"=="5" (
    python scripts\test_cuda.py
) else if "%choice%"=="0" (
    exit
) else (
    echo Invalid choice
)
pause
