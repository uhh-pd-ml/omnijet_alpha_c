#!/bin/bash
# NOTE: This script is an example and should be adjusted to your needs.
# The fields which need to be adjusted are marked with "ADJUST THIS".

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu
#SBATCH --constraint="A100&GPUx1"
#SBATCH --time=100:00:00
#SBATCH --exclude=
#SBATCH --job-name TokenGen_example
#SBATCH --output /data/dust/user/rosehenn/gabbro_output/logs/slurm_logs/%x_%j.log      # ADJUST THIS to your log path
#SBATCH --mail-user your.email

echo "Starting job $SLURM_JOB_ID with the following script:"
echo "----------------------------------------------------------------------------"
echo
cat $0

source ~/.bashrc
cd /data/dust/user/rosehenn/gabbro/  # ADJUST THIS to your repository path

LOGFILE="/data/dust/user/rosehenn/gabbro_output/logs/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"  # ADJUST THIS to your log path
PYTHON_COMMAND="python gabbro/train.py experiment=example_experiment_backbone_generative.yaml"

# run the python command in the singularity container
# ADJUST THIS to your singularity image path
singularity exec --nv --bind /data:/data \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" --env SLURM_LOGFILE="$LOGFILE" \
    docker://jobirk/omnijet:latest \
    bash -c "source /opt/conda/bin/activate && $PYTHON_COMMAND"
## ---------------------- End of job script -----------------------------------
################################################################################
