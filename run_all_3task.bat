@echo off
echo ========================================
echo 3-Task Training: All Optimizers
echo ========================================
echo.

echo [1/5] Training SGD...
python train_3task.py --optimizer sgd --batch-size 16 --epochs 5 --lr 0.01

echo.
echo [2/5] Training RMSprop...
python train_3task.py --optimizer rmsprop --batch-size 16 --epochs 5 --lr 0.001

echo.
echo [3/5] Training Adam...
python train_3task.py --optimizer adam --batch-size 16 --epochs 5 --lr 0.001

echo.
echo [4/5] Training Muon...
python train_3task.py --optimizer muon --batch-size 16 --epochs 5 --lr 0.02

echo.
echo [5/5] Training MTMuon...
python train_3task.py --optimizer mtmuon --batch-size 16 --epochs 5 --lr 0.02

echo.
echo ========================================
echo ALL TRAINING COMPLETE!
echo ========================================
pause
