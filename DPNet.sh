#!/bin/bash
#SBATCH --job-name=dpnetkpt
#SBATCH --output=out/kptrcnn_E.out
#SBATCH --error=out/kptrcnn_E.err

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1

#SBATCH --time=23:59:59

source /home/shernandez/.bashrc
module load CUDA/11.4.3
conda activate CycleDetect_env

python -u projects/DopplerNet/train.py --config_file /home/shernandez/projects/DopplerNet/files/Train.yaml > out/kptrcnn_E.out