#!/bin/bash
#SBATCH --job-name=qwen_32B_eval_counting_sa
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --output=/path/to/slurm/logs/output/qwen_32B_eval_counting_sa-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/qwen_32B_eval_counting_sa-%j.txt
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
    --experiment_variant "single_agent+both_views" \
    --single_agent_model_name "Qwen/Qwen3-VL-32B-Instruct" \
    --single_agent_client_name "vllm" \
    --single_agent_api_base "${ANSWERER_API_BASE:-http://localhost:4877/v1}"
