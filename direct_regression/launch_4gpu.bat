@echo off
REM Launch 4 GPU distributed training for Enhanced Direct Model

echo ============================================================
echo Enhanced Direct Regression Model - 4 GPU Training
echo ============================================================
echo.

REM Check CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
echo.

REM Set environment variables for optimal performance
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

REM Run distributed training
echo Starting training on 4 GPUs...
echo.

python train_enhanced_4gpu.py

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
