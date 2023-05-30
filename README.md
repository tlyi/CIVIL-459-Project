# Pifpaf Lane Detection
Project description
--------------------
This project is part of EPFL Deep Learning for Autonomous Vehicles class, aiming to solve one sub-task related to autonomous driving: lane detection. Despite the state-of-the-art 3d lane detection models like [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane) using anchor-based perspective transformer to transform front-eye-view features to bird-eye view features, we are inspired by the idea of [openpifpaf](https://openpifpaf.github.io/intro.html), which achieves human pose-estimation by detecting and associating spatial-temporal human joint keypoints. Using the same idea, we are trying here to simplify the task of detecting and regressing potentially hundreds or even thousands of pixel points of road lane to a few key points and connect them to form an estimate of a lane, which may significantly reduce the time required for lane detection. 

Essentially, our contribution is to extend the function of [openpifpaf](https://openpifpaf.github.io/intro.html) to lane detection by enabling it to be trained on a whole different dataset:[OpenLane](https://github.com/OpenDriveLab/OpenLane).
