#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023

#SBATCH --time 3-00:00:00

echo STARTING


python3 -m openpifpaf.train --lr=0.002 --momentum=0.9 --b-scale=5.0 --debug \
  --epochs=1000 \
  --lr-warm-up-factor=0.25 \
  --output=outputs/2kps/2kps_10percent \
  --batch-size=5  --val-batches=1 --val-interval=10 \
  --weight-decay=1e-5 \
  --dataset=openlane --openlane-no-augmentation \
  --basenet=shufflenetv2k16 \
  --openlane-train-annotations data_openlane_2kps/annotations/openlane_keypoints_sample_training.json \
  --openlane-val-annotations data_openlane_2kps/annotations/openlane_keypoints_sample_validation.json \
  --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
  --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \
  --loader-workers 1