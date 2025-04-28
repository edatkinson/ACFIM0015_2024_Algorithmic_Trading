#!/bin/bash
#SBATCH --job-name=marketsim
#SBATCH –-account=emat024603
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1                       #depends on the number of yaml files
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # Adjust based on your workload
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-12
#SBATCH --partition=compute
#SBATCH --mail-user=wd21585@bristol.ac.uk


source algoenv/bin/activate

# Get the list of YAML files
YAML_FILES=($(ls yaml_files/*.yaml))
YAML=${YAML_FILES[$SLURM_ARRAY_TASK_ID]} # take each yaml file by index

WORKDIR=work/

echo "Running simulation for YAML file: $YAML"
echo "Output Data to: $WORKDIR"
python3 run_session.py "$YAML" "$WORKDIR" 
