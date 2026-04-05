#!/bin/bash

# Load environment variables
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Download COSMIC benchmark data from HuggingFace
python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download

data_dir = os.environ.get("DATA_DIR", "./data")
os.makedirs(data_dir, exist_ok=True)

print(f"Downloading COSMIC dataset to {data_dir} ...")
snapshot_download(
    repo_id="mair-lab/Cosmic",
    repo_type="dataset",
    local_dir=data_dir,
    token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
)
print("Done.")

# List downloaded JSON files
json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
if json_files:
    print("\nDownloaded JSON files:")
    for f in sorted(json_files):
        print(f"  {f}")
EOF
