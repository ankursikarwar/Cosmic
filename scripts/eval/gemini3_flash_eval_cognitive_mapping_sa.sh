#!/bin/bash
#SBATCH --job-name=gemini3_flash_eval_map_sa
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --output=/path/to/slurm/logs/output/gemini3_flash_eval_map_sa-%j.txt
#SBATCH --error=/path/to/slurm/logs/error/gemini3_flash_eval_map_sa-%j.txt
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
    --tasks_qa_file "${DATA_DIR:-./data}/cognitive_mapping/test-00000.parquet" \
    --experiment_variant "single_agent+both_views" \
    --single_agent_model_name "gemini-3-flash-preview" \
    --single_agent_client_name "gemini" \
    --single_agent_api_base "https://generativelanguage.googleapis.com/v1beta/openai/"
