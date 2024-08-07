#!/bin/bash
#SBATCH --job-name=in9-bg-reliance                            # Specify a name for your job
#SBATCH --output=slurm-logs/out-in9-bg-reliance-%j.log        # Specify the output log file
#SBATCH --error=slurm-logs/err-in9-bg-reliance-%j.log         # Specify the error log file
#SBATCH --nodes=1                                             # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                                     # Number of CPU cores per task
#SBATCH --time=04:00:00                                       # Maximum execution time (HH:MM:SS)
#SBATCH --qos=default                                         # Specify the partition (queue) you want to use
#SBATCH --gres=gpu:rtxa5000:1                                 # Number of GPUs per node
#SBATCH --mem=32G                                             # Memory per node


# Project setup
proj_root="/vulcanscratch/mhoover4/code/fmm-attention"

# Load any required modules or activate your base environment here if necessary
source $proj_root/.venv/bin/activate

python $proj_root/examples/run_image_example.py
