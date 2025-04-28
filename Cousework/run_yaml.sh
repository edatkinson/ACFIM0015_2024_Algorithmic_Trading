#!/bin/bash
#SBATCH --job-name=marketsim
#SBATCH --account=emat024603
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --partition=compute

source ~/algoenv/bin/activate

DIR=work/ #work directory in HPC

echo "Running simulation for YAML file: $1"
echo "Output Data to: $DIR"

python3 run_session.py "$1" "$DIR"

# Submits one yaml scenario individually
# sbatch --nodelist=bp1-compute031 run_yaml.sh yaml_files/template.yaml
# sbatch --nodelist=bp1-compute032 run_yaml.sh yaml_files/test.yaml
