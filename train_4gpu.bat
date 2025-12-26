@echo off
REM Training script for 4 A100 GPUs (80GB each) - Windows version

SET CONFIG=config/multi_view_config.json
SET CHECKPOINT_DIR=checkpoints_4gpu
SET NUM_GPUS=4
SET METHOD=accelerate
SET USE_WANDB=

REM Parse arguments
:parse_args
IF "%~1"=="" GOTO end_parse
IF "%~1"=="--config" (
    SET CONFIG=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--checkpoint_dir" (
    SET CHECKPOINT_DIR=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--method" (
    SET METHOD=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--wandb" (
    SET USE_WANDB=--wandb
    SHIFT
    GOTO parse_args
)
SHIFT
GOTO parse_args

:end_parse

echo ==========================================
echo Training Hybrid-ViT Cascade on 4 A100 GPUs
echo ==========================================
echo Config: %CONFIG%
echo Checkpoint Dir: %CHECKPOINT_DIR%
echo Method: %METHOD%
echo ==========================================

IF "%METHOD%"=="torchrun" (
    echo Using torchrun with DDP...
    torchrun --nproc_per_node=%NUM_GPUS% --master_port=29500 training/train_distributed.py --config %CONFIG% --checkpoint_dir %CHECKPOINT_DIR% %USE_WANDB%
)

IF "%METHOD%"=="accelerate" (
    echo Using Accelerate...
    accelerate launch --multi_gpu --num_processes=%NUM_GPUS% --mixed_precision=fp16 training/train_accelerate.py --config %CONFIG% --checkpoint_dir %CHECKPOINT_DIR% %USE_WANDB%
)

echo Training completed!
pause
