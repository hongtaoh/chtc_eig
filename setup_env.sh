#!/bin/bash
set -e

# Initialize conda for this script
source ~/miniconda/etc/profile.d/conda.sh

conda deactivate 2>/dev/null || true
conda remove --name llm --all -y 2>/dev/null || true

conda create -n llm python=3.10 -y
conda activate llm
conda install -c conda-forge conda-pack -y

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate sentencepiece safetensors pandas PyYAML numpy matplotlib 

python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}')"

conda-pack -n llm --output /staging/hhao9/llm.tar.gz --force

echo "Done! Env at /staging/hhao9/llm.tar.gz"