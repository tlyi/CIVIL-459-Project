#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH --time 3-00:00:00

python3 -m openpifpaf.eval \
--dataset=openlane --loader-workers=1 \
--checkpoint outputs/2kps/2kps_10percent_2ndrun.epoch079 \
--seed-threshold=0.2 \
--openlane-train-annotations data_openlane_2kps/annotations/openlane_keypoints_sample_training.json \
--openlane-val-annotations data_openlane_2kps/annotations/openlane_keypoints_sample_validation.json \
--openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
--openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \
--output eval/2kps_meanpixelerror_print3