@echo off
echo ========================================
echo 5-HOUR TRAINING PLAN
echo ========================================
echo.
echo Configuration:
echo - 100 classes (ImageNet)
echo - 21 classes (COCO segmentation)
echo - 50,000 train + 10,000 val images
echo - Batch size: 32
echo - Epochs: 10
echo.
echo Training 4 optimizers (~40 min each):
echo.

echo [1/4] Training SGD...
python train.py --optimizer sgd --batch-size 32 --epochs 10 --lr 0.01

echo.
echo [2/4] Training Adam...
python train.py --optimizer adam --batch-size 32 --epochs 10 --lr 0.001

echo.
echo [3/4] Training AdamW...
python train.py --optimizer adamw --batch-size 32 --epochs 10 --lr 0.001

echo.
echo [4/4] Training MTAdamV2...
python train.py --optimizer mtadamv2 --batch-size 32 --epochs 10 --lr 0.0003

echo.
echo ========================================
echo ALL TRAINING COMPLETE!
echo Check results/ folder for outputs
echo ========================================
pause
