#!/bin/bash
# run.sh - Submit jobs to CHTC
# Automatically calculates number of jobs based on config.yaml

set -e

# Optional: remove previous jobs
condor_rm hhao9 2>/dev/null || true

echo "Cleaning old logs..."
rm -rf log/
rm -rf results/

# Read config
TOTAL=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['job_settings']['total_patients'])")
BATCH=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['job_settings']['batch_size'])")

# Calculate number of jobs needed (ceiling division)
NUM_JOBS=$(( (TOTAL + BATCH - 1) / BATCH ))

echo "========================================"
echo "Configuration:"
echo "  Total patients: $TOTAL"
echo "  Batch size: $BATCH"
echo "  Number of jobs: $NUM_JOBS"
echo "========================================"

echo "Submitting $NUM_JOBS jobs..."
condor_submit gpu-lab.sub -a "queue $NUM_JOBS"

echo "Done! Monitor with: condor_q"