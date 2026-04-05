#!/bin/bash
#SBATCH --job-name=qwen_8B_eval_relative
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --output=/path/to/slurm/logs/output/qwen_8B_eval_relative-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/qwen_8B_eval_relative-%j.txt
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
    --tasks_qa_file "${DATA_DIR:-./data}/relative_distance/test-00000.parquet" \
    --terminate \
    --experiment_variant "two_agent+parallel" \
    --max_num_turns 10 \
    --answerer_model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --helper_model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --answerer_api_base "${ANSWERER_API_BASE:-http://localhost:4877/v1}" \
    --helper_api_base "${HELPER_API_BASE:-http://localhost:4877/v1}"
