@echo off
echo ========================================
echo Setting up CUDA Environment
echo ========================================
echo.

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/4] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/4] Installing other requirements...
pip install numpy pillow matplotlib tqdm pycocotools tabulate

echo.
echo ========================================
echo Testing CUDA availability...
echo ========================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate environment: venv\Scripts\activate
echo To train: python train.py --optimizer adam --batch-size 8 --epochs 2
echo.
pause
