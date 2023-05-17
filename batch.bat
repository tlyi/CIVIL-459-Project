#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023

echo STARTING

python3 -m openpifpaf.train \
    --dataset openlane \
    --openlane-train-annotations ../data_openlane/annotations/openlane_keypoints_sample_training.json \
    --openlane-val-annotations ../data_openlane/annotations/openlane_keypoints_sample_validation.json \
    --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
    --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \
    --openlane-square-edge=769 \
    --basenet=shufflenetv2k16 --lr=0.00002 --momentum=0.95  --b-scale=5.0 \
    --epochs=300 --lr-decay 160 260 --lr-decay-epochs=10  --weight-decay=1e-5 \
    --weight-decay=1e-5  --val-interval 10 --loader-workers 16 --openlane-upsample 2 \
    --openlane-bmin 2 --batch-size 8