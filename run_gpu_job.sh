#!/bin/bash
set -e

# Arguments from Submit File
CLUSTER=$1
PROCESS=$2

echo "========================================"
echo "CHTC Medical Diagnosis Experiment"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Cluster: $CLUSTER, Process: $PROCESS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "========================================"

export PYTHONNOUSERSITE=1

# --- 1. READ CONFIG ---
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model_name'])")
BATCH_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['job_settings']['batch_size'])")

echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"

# Staging paths
ENV_TARBALL="/staging/hhao9/llm.tar.gz"
MODEL_TARBALL="/staging/hhao9/my_models/${MODEL_NAME}.tar.gz"

# --- 2. EXTRACT ENVIRONMENT ---
echo "Extracting conda environment from staging..."
if [ ! -d ".conda_env" ]; then
    mkdir -p .conda_env
    tar -xzf "$ENV_TARBALL" -C .conda_env
fi

# --- 3. EXTRACT MODEL ---
echo "Extracting model from staging..."
tar -xzf "$MODEL_TARBALL"

PYTHON_EXEC="$(pwd)/.conda_env/bin/python"

# Setup HF
export HF_HOME="$(pwd)/hf_home"
export HF_OFFLINE=1
mkdir -p "$HF_HOME"

# Verify setup
echo "Verifying environment..."
"$PYTHON_EXEC" -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# --- 4. CALCULATE INDICES ---
# PROCESS is 0-indexed
START_IDX=$((PROCESS * BATCH_SIZE))
END_IDX=$((START_IDX + BATCH_SIZE))

# Job ID is 1-indexed for human readability
JOB_ID=$((PROCESS + 1))

echo "Job $JOB_ID: Processing patients [$START_IDX, $END_IDX)"

# Create results directory
mkdir -p results

# --- 5. RUN EXPERIMENT ---
"$PYTHON_EXEC" main.py \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --model_path "./${MODEL_NAME}" \
    --job_id $JOB_ID

echo "========================================"
echo "Job completed at $(date)"
echo "========================================"