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

@REM python3 -m openpifpaf.train \
@REM     --dataset openlane \
@REM     --openlane-train-annotations data_openlane/annotations/openlane_keypoints_sample_training.json \
@REM     --openlane-val-annotations data_openlane/annotations/openlane_keypoints_sample_validation.json \
@REM     --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
@REM     --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \
@REM     --openlane-square-edge=769 \
@REM     --basenet=shufflenetv2k16 --lr=0.00002 --momentum=0.95  --b-scale=5.0 \
@REM     --epochs=300 --lr-decay 160 260 --lr-decay-epochs=10  --weight-decay=1e-5 \
@REM     --weight-decay=1e-5  --val-interval 10 --loader-workers 16 --openlane-upsample 2 \
@REM     --openlane-bmin 2 --batch-size 8

  
python3 -m openpifpaf.train \
  --lr=0.01 --momentum=0.9 --b-scale=5.0 \
  --epochs=10 --lr-warm-up-epochs=10 \
  --batch-size=4 --train-batches=1 --val-batches=1 --val-interval=100 \
  --weight-decay=1e-5 \
  --dataset=openlane  --basenet=shufflenetv2k16 --debug