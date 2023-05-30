#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH --time 3-00:00:00

python3 -m openpifpaf.eval \
--dataset=openlane --loader-workers=1 \
--checkpoint outputs/shufflenetv2k16-230529-170501-openlane-slurm1381136.pkl.epoch038 \
--force-complete-pose --seed-threshold=0.2
