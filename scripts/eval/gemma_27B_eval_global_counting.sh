#!/bin/bash
#SBATCH --job-name=gemma_27B_eval_counting
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --output=/path/to/slurm/logs/output/gemma_27B_eval_counting-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/gemma_27B_eval_counting-%j.txt
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=64Gb

# Load required modules
module load cuda/12.2.2
module load anaconda/3
conda activate infinigen

# Load environment variables
ENV_FILE="$(dirname "$0")/../../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Run from project root
cd "$(dirname "$0")/../.."

python3 main.py \
    --tasks_qa_file "${DATA_DIR:-./data}/global_counting/test-00000.parquet" \
    --terminate \
    --experiment_variant "two_agent+parallel" \
    --max_num_turns 10 \
    --answerer_model_name "google/gemma-3-27b-it" \
    --helper_model_name "google/gemma-3-27b-it" \
    --answerer_api_base "${ANSWERER_API_BASE:-http://localhost:4877/v1}" \
    --helper_api_base "${HELPER_API_BASE:-http://localhost:4877/v1}"
