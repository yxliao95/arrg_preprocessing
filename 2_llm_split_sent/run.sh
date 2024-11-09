#!/bin/bash

#SBATCH --job-name=test_llm
#SBATCH --account=scw2258

# job stdout file. The '%J' to Slurm is replaced with the job number.
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_sentgen/outputs/logs/std/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_sentgen/outputs/logs/std/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:1
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cuda=CUDA/12.4
conda=anaconda/2024.06
env=arrg_sentgen

module load $cuda
echo "Loaded $cuda"

module load $conda
source activate
conda activate $env
echo "Loaded $conda, env: $env"
nvcc -V

python /scratch/c.c21051562/workspace/arrg_sentgen/llm_split_sent.py --from_bash --partition 3
# python /scratch/c.c21051562/workspace/arrg_sentgen/test.py

python /scratch/c.c21051562/workspace/arrg_sentgen/test_email.py

# nohup/scratch/c.c21051562/workspace/arrg_cxrgraph/run_joint.sh > nohup.log 2>&1 &

# sbatch /scratch/c.c21051562/workspace/arrg_sentgen/basic.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID