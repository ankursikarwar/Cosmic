#!/bin/bash
#SBATCH --job-name=vllm_server_internvl_35_8B
#SBATCH --partition=long
#SBATCH -c 24
#SBATCH --output=/path/to/slurm/logs/output/vllm_server_job_output-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/vllm_server_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --time=14:00:00
#SBATCH --mem=256Gb

# Load required modules
module load cuda/12.2.2
module load anaconda/3
conda activate vllm_new

# Load environment variables
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi


# Set environment variables
export HF_HOME=$HOME/scratch/cache

# Save hostname to file
PROJECT_DIR=$HOME/Work/MultiAgent
mkdir -p $PROJECT_DIR
hostname > $PROJECT_DIR/vllm_server_node.txt

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Launch vllm server
export VLLM_USE_V1=1

vllm serve OpenGVLab/InternVL3_5-8B-Instruct \
    --port 4877 \
    --host 0.0.0.0 \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization 0.90 \
    --mm-encoder-tp-mode data \
    --dtype auto \
    --async-scheduling \
    --limit-mm-per-prompt.video 0 \
    --trust-remote-code \
    # --disable-mm-preprocessor-cache
    # --max-num-seqs 256 \
    # --max-model-len 32768 \
