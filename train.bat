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


echo STARTING

python3 -m openpifpaf.train --lr=0.001 --momentum=0.9 --b-scale=5.0 --debug \ 
  --epochs=10000 \ 
  --lr-warm-up-factor=0.3 \ 
  --batch-size=4  --val-batches=1 --val-interval=100 \ 
  --weight-decay=1e-5 \ 
  --dataset=openlane --openlane-upsample=2 --openlane-no-augmentation \ 
  --basenet=shufflenetv2k16 \ 
  --openlane-train-annotations data_openlane_3/annotations/openlane_keypoints_sample_training.json \ 
  --openlane-val-annotations data_openlane_3/annotations/openlane_keypoints_sample_validation.json \ 
  --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \ 
  --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \ 
  --loader-workers 1