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
#SBATCH --reservation civil-459

echo STARTING

python3 -m openpifpaf.train --dataset openlane --openlane-train-annotations data_openlane/annotations/openlane_keypoints_sample_10training.json --openlane-val-annotations data_openlane/annotations/openlane_keypoints_sample_10validation.json  --openlane-train-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training --openlane-val-image-dir /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation --openlane-square-edge=769  --checkpoint outputs/shufflenetv2k16-230521-221626-openlane-slurm1374997.pkl.epoch010   --lr=0.001 --momentum=0.9  --b-scale=5.0 --epochs=100 --weight-decay=1e-5  --val-interval 10 --loader-workers 1 --batch-size 8  --lr-decay-factor=0.2 --lr-warm-up-epochs=1 