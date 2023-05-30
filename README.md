# Pifpaf Lane Detection
Project description
--------------------
This project is part of EPFL Deep Learning for Autonomous Vehicles class, aiming to solve one sub-task related to autonomous driving:
lane detection. Despite the state-of-the-art 3d lane detection models like [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane) 
using anchor-based perspective transformer to transform front-eye-view features to bird-eye view features, we are inspired by the idea of [openpifpaf](https://openpifpaf.github.io/intro.html), which achieves human pose-estimation by detecting and associating spatial-temporal human joint keypoints.
Using the same idea, we are trying here to simplify the task of detecting and regressing potentially hundreds or even thousands of pixel points of road lane to a few key points and connect them to form an estimate of a lane, which may significantly reduce the time required for lane detection. 

Essentially, our contribution is to extend the function of [openpifpaf](https://openpifpaf.github.io/intro.html) to lane detection by enabling it to be trained 
on a whole different dataset: [OpenLane](https://github.com/OpenDriveLab/OpenLane). We transformed OpenLane dataset to CoCo format and downsampled original
lane annotation to several keypoints (24 and 2, respectively). Plugin necessities were implemented without changing the main body of openpifpaf,
making it easy to comply with the original model. To note that due to the very different nature of the two datasets plus time and resource limit, 
our progress is currently mainly on 2d lane detection with massive debugging, explorations, trial and error, but the preliminary results show the feasibility 
of this idea and clear way to 3d lane detection extension.

Dataset description
-------------------
Description of the dataset + label format + where/how to acquire it. What data do I need
to train your model? How do I get it? In what shape?

Installation
-------------
how to install this package and original openpifpaf, any modifications needed for original openpifpaf?

Code
------
Train, predict, evaluation, flag arguments

Experimental setup
------------------
What are the experiments you conducted? What are the evaluation
metrics?

Results
--------
**24 keypoints**

**2 keypoints (closest and most far away)**

Conclusion
----------
Short one

