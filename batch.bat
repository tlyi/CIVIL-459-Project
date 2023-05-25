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

!/bin/bash
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

python3 -m openpifpaf.train --dataset openlane \
                            --openlane-train-annotations data_openlane_3/annotations/openlane_keypoints_sample_training.json \
                            --openlane-val-annotations data_openlane_3/annotations/openlane_keypoints_sample_validation.json \
                            --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training \
                            --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation \
                            --openlane-square-edge=769  --lr=0.001 --momentum=0.9  --b-scale=5.0 \
                            --epochs=100 --lr-decay 160 260 --lr-decay-epochs=5  --weight-decay=1e-5 --openlane-bmin 2  \
                            --weight-decay=1e-5  --val-interval 10 --loader-workers 1 --batch-size 4  --lr-decay-factor=0.2 --lr-warm-up-epochs=1 \
                            --checkpoint outputs/shufflenetv2k16-230521-171516-openlane-slurm1374922.pkl.epoch000 
