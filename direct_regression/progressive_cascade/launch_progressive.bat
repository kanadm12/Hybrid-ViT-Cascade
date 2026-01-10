@echo off
REM Progressive Cascade Training Launcher
REM Automatically uses all available GPUs for 4-GPU distributed training

echo ========================================
echo Progressive Multi-Scale CT Reconstruction
echo 64^3 -^> 128^3 -^> 256^3 Cascade
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python or activate your conda environment
    pause
    exit /b 1
)

REM Optional: Activate conda environment
REM Uncomment and modify the line below if using conda
REM call conda activate your_env_name

REM Optional: Set specific GPUs to use
REM Uncomment to use only specific GPUs (e.g., GPU 0,1,2,3)
REM set CUDA_VISIBLE_DEVICES=0,1,2,3

echo Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo.

echo Configuration:
echo   - Stage 1: 64^3 volume, 50 epochs, batch_size=8
echo   - Stage 2: 128^3 volume, 30 epochs, batch_size=4  
echo   - Stage 3: 256^3 volume, 20 epochs, batch_size=2
echo.
echo Starting progressive training...
echo.

REM Run training with 4-GPU DDP
python train_progressive_4gpu.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo Training Failed!
    echo ========================================
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Checkpoints saved to: checkpoints_progressive/
echo   - stage1_best.pth (64^3 model)
echo   - stage2_best.pth (64^3+128^3 model)
echo   - stage3_best.pth (full 64^3+128^3+256^3 model)
echo.
echo Next steps:
echo   1. Run inference: python inference_progressive.py --checkpoint checkpoints_progressive/stage3_best.pth
echo   2. Evaluate on test set: python inference_progressive.py --mode evaluate
echo.
pause
