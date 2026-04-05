#!/bin/bash
#SBATCH --job-name=gemini3_pro_eval_anchor
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --output=/path/to/slurm/logs/output/gemini3_pro_eval_anchor-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/gemini3_pro_eval_anchor-%j.txt
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=64Gb

# Load required modules
module load cuda/12.2.2
module load anaconda/3
conda activate cosmic

# Load environment variables
ENV_FILE="$(dirname "$0")/../../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Run from project root
cd "$(dirname "$0")/../.."

python3 main.py \
    --tasks_qa_file "${DATA_DIR:-./data}/anchor_recognition/test-00000.parquet" \
    --terminate \
    --experiment_variant "two_agent+parallel" \
    --max_num_turns 10 \
    --answerer_model_name "gemini-3-pro-preview" \
    --helper_model_name "gemini-3-pro-preview" \
    --answerer_client_name "gemini" \
    --helper_client_name "gemini" \
    --answerer_api_base "https://generativelanguage.googleapis.com/v1beta/openai/" \
    --helper_api_base "https://generativelanguage.googleapis.com/v1beta/openai/"
