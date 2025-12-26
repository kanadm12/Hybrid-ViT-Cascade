#!/bin/bash

# RunPod Setup and Training Script for Hybrid-ViT Cascade
# This script automates the setup process on a RunPod GPU instance

set -e  # Exit on error

echo "======================================================================"
echo "Hybrid-ViT Cascade - RunPod Setup and Training"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Check environment
echo "Step 1: Checking environment..."
if command -v nvidia-smi &> /dev/null; then
    print_status "CUDA/GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "No GPU detected. Please ensure you're on a GPU pod."
    exit 1
fi

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    print_status "Python detected: $PYTHON_VERSION"
else
    print_error "Python not found"
    exit 1
fi

# Step 2: Navigate to project directory
echo ""
echo "Step 2: Setting up project directory..."

# Try to find the project
PROJECT_PATHS=(
    "/workspace/x2ctpa/hybrid_vit_cascade"
    "/workspace/hybrid_vit_cascade"
    "./hybrid_vit_cascade"
    "."
)

PROJECT_DIR=""
for path in "${PROJECT_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/requirements.txt" ]; then
        PROJECT_DIR="$path"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    print_error "Project directory not found. Please ensure code is uploaded."
    exit 1
fi

cd "$PROJECT_DIR"
print_status "Project directory: $PROJECT_DIR"

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q nibabel scipy Pillow wandb

print_status "Dependencies installed"

# Step 4: Verify data
echo ""
echo "Step 4: Verifying data directory..."

DATA_PATHS=(
    "workspace/drr_patient_data"
    "./workspace/drr_patient_data"
    "/workspace/drr_patient_data"
    "../workspace/drr_patient_data"
)

DATA_DIR=""
for path in "${DATA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        DATA_DIR="$path"
        break
    fi
done

if [ -z "$DATA_DIR" ]; then
    print_warning "Data directory not found in standard locations."
    echo "Searched in:"
    for path in "${DATA_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    read -p "Enter the path to your drr_patient_data directory (or press Enter to skip): " CUSTOM_DATA_PATH
    if [ -n "$CUSTOM_DATA_PATH" ] && [ -d "$CUSTOM_DATA_PATH" ]; then
        DATA_DIR="$CUSTOM_DATA_PATH"
    else
        print_error "Data directory not accessible. Please upload your data and try again."
        exit 1
    fi
fi

# Count patient folders
PATIENT_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type d -name "patient_*" | wc -l)
print_status "Data directory: $DATA_DIR"
print_status "Found $PATIENT_COUNT patient folders"

# Update config with actual data path
CONFIG_FILE="config/runpod_config.json"
if [ -f "$CONFIG_FILE" ]; then
    # Use Python to update the config properly
    python << EOF
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['data']['dataset_path'] = '$DATA_DIR'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print("Updated config with data path")
EOF
    print_status "Config updated with data path"
fi

# Step 5: Setup W&B (optional)
echo ""
echo "Step 5: Weights & Biases setup (optional)..."
read -p "Do you want to use W&B for logging? (y/n): " USE_WANDB

WANDB_FLAG=""
if [ "$USE_WANDB" = "y" ] || [ "$USE_WANDB" = "Y" ]; then
    if command -v wandb &> /dev/null; then
        read -p "Enter your W&B API key (or press Enter if already logged in): " WANDB_KEY
        if [ -n "$WANDB_KEY" ]; then
            wandb login "$WANDB_KEY"
        fi
        WANDB_FLAG="--wandb"
        read -p "Enter W&B project name (default: xray2ct-hybrid-vit): " WANDB_PROJECT
        WANDB_PROJECT=${WANDB_PROJECT:-xray2ct-hybrid-vit}
        WANDB_FLAG="$WANDB_FLAG --wandb_project $WANDB_PROJECT"
        print_status "W&B configured"
    else
        print_warning "W&B not installed, skipping"
    fi
fi

# Step 6: Create checkpoint directory
echo ""
echo "Step 6: Setting up checkpoint directory..."
CHECKPOINT_DIR="/workspace/checkpoints/hybrid_vit_cascade"
mkdir -p "$CHECKPOINT_DIR"
print_status "Checkpoint directory: $CHECKPOINT_DIR"

# Step 7: Display training configuration
echo ""
echo "======================================================================"
echo "Training Configuration Summary"
echo "======================================================================"
echo "Project Directory:    $PROJECT_DIR"
echo "Data Directory:       $DATA_DIR"
echo "Patient Count:        $PATIENT_COUNT"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "W&B Logging:          $([ -n "$WANDB_FLAG" ] && echo 'Enabled' || echo 'Disabled')"
echo "Config File:          $CONFIG_FILE"
echo "======================================================================"
echo ""

# Step 8: Ask to start training
read -p "Start training now? (y/n): " START_TRAINING

if [ "$START_TRAINING" != "y" ] && [ "$START_TRAINING" != "Y" ]; then
    print_status "Setup complete. Run training manually with:"
    echo ""
    echo "  cd $PROJECT_DIR"
    echo "  python training/train_runpod.py --config config/runpod_config.json $WANDB_FLAG"
    echo ""
    exit 0
fi

# Step 9: Start training
echo ""
echo "======================================================================"
echo "Starting Training..."
echo "======================================================================"
echo ""

# Run training with or without W&B
python training/train_runpod.py \
    --config config/runpod_config.json \
    $WANDB_FLAG \
    --checkpoint_dir "$CHECKPOINT_DIR"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    print_status "Training completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Checkpoints saved to: $CHECKPOINT_DIR"
    ls -lh "$CHECKPOINT_DIR"
else
    print_error "Training failed. Check the error messages above."
    exit 1
fi
