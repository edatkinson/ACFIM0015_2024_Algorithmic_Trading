#!/bin/bash 

# Script which sets up testing from root directory


YAML_DIR="./yaml_files"
SCRIPT="run_session.py"

source algoenv/bin/activate

# Loop through all YAML files

for yaml_file in "$YAML_DIR"/*.yaml; do
    echo "====================================="
    echo "Running simulation for: $yaml_file"
    echo "====================================="
    python "$SCRIPT" "$yaml_file"

done



