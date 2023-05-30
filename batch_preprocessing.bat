#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023


echo STARTING

python3 -m openpifpaf_openlane.openlane_to_coco_2kp \
    --dir_data='/work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/lane3d_300' \
    --dir_images='/work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images' \
    --dir_out='./data_openlane_2kps' \
    --sample \
