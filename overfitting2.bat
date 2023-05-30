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


python3 -m openpifpaf.train \
  --dataset openlane \
  --openlane-train-annotations data_openlane/annotations/openlane_keypoints_sample_10training.json \
  --openlane-val-annotations data_openlane/annotations/openlane_keypoints_sample_10validation.json \
  --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
  --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation --openlane-square-edge 769 --basenet=shufflenetv2k16 \
  --lr=0.0001 --momentum=0.95 --b-scale=15.0 \
  --epochs=1000 --lr-warm-up-epochs=100 \
  --batch-size=4 --train-batches=1 --val-batches=1 --val-interval=100 \
  --weight-decay=1e-5 --clip-grad-value=10 --lr-decay 120 140 --lr-decay-epochs=10 \
  --debug
