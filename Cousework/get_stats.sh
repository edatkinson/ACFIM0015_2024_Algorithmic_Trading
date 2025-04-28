#!/bin/bash
#SBATCH --job-name=get_stats
#SBATCH --account=emat024603
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=compute



source ~/algoenv/bin/activate

FILENAME=$1 # name of the file to be saved

python3 analyse_performance.py --yaml_file yaml_files/GoldenStandard.yaml --dir work/ --f "$FILENAME"


