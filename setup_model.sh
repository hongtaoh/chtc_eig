#!/bin/bash
set -e

# MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME="Qwen-0.5B"

MODEL_REPO=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model_repo'])")
MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model_name'])")

cd /staging/hhao9/my_models

rm -rf "$MODEL_NAME" "${MODEL_NAME}.tar.gz"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_REPO',
    local_dir='$MODEL_NAME'
)
"

tar -czf "${MODEL_NAME}.tar.gz" "$MODEL_NAME"
rm -rf "$MODEL_NAME"

echo "Done! Model at /staging/hhao9/my_models/${MODEL_NAME}.tar.gz"