# OpenPifPaf Lane Detection
<p align="center" width="100%">
    <img width="100%" src="https://github.com/tlyi/CIVIL-459-Project/assets/69505852/35edc49b-4afe-48e3-af65-32740b9c46c7">
</p>

## Quick Navigation
- [Project Description](#project-description)
- [Contribution Overview](#contribution-overview)
- [Installation](#installation)
- [Dataset Description](#dataset-description)
- [Code](#code)
  - [Train](#train)
  - [Predict](#predict)
  - [Evaluate](#evaluate)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Further Improvements](#further-improvements)
- [Conclusion](#conclusion)
- [References](#references)


## Project Description
This project is part of EPFL's "Deep Learning for Autonomous Vehicles" course. 
This year, the final goal of the course is to build the main computer vision components of an autonomous vehicle. This project aims to contribute to one small but important module of this big system, 3D lane detection. We are inspired by the idea of [OpenPifPaf](https://openpifpaf.github.io/intro.html), which achieves human pose-estimation by detecting and associating spatial-temporal human joint keypoints.

We see the potential in using the same concept to simplify the task of detecting and regressing potentially hundreds or even thousands of pixel points of road lane to just a few key points and connect them to form an estimate of a lane. We hope that this opens up hope for a new framework that can significantly reduce the time required for lane detection.

## Contribution Overview

In summary, our contribution is to extend the function of [OpenPifPaf](https://openpifpaf.github.io/intro.html) by creating a plugin for it that enables it to perform lane detection. This would enable it to be trained and evaluated on a whole different dataset: [OpenLane](https://github.com/OpenDriveLab/OpenLane). Plugin necessities were implemented without changing the main body of OpenPifPaf, making the project easy to install and set up. 

Due to the very different nature of the two datasets, coupled with time and resource limits, our progress is currently mainly on 2D lane detection. However, the preliminary results prove the feasibility of this idea and opens up the possibility of extension to 3D lane detection. 

## Installation
### 1. Clone this repository
``` bash
git clone https://github.com/tlyi/CIVIL-459-Project.git
```

### 2. Install OpenPifPaf
This will also install all the required dependencies. We recommend doing this in a virtual environment like Anaconda. 
``` bash
pip3 install openpifpaf
```

OpenPifPaf uses a Plugin architectural pattern which allows us to train it on a custom dataset without having to alter the core network.  

The required files to register our dataset as a plugin is contained in the folder `openpifpaf_openlane`. (IMPORTANT: Do not change the name of this folder as OpenPifPaf requires this naming convention to recognise the plugin.)

### 3. Download checkpoint (optional)
The dataset that we are using is very big and will take days to train. We have provided a [checkpoint](https://drive.google.com/file/d/1IEKkXFKS5HWgyEEhrRoDiCdvLgZRtmV7/view?usp=sharing) that has already been trained on 30 epochs on 10% of the dataset, which already took about 5 days. You may choose to train either from scratch (not recommended), or from one of the backbones provided by OpenPifPaf, or on top of our provided checkpoint. 

## Dataset Description
The dataset that we have chosen to work with is [OpenLane](https://github.com/OpenDriveLab/OpenLane). OpenLane is the largest scale real world 3D lane dataset. It owns 200K frames and over 880K carefully annotated lanes, where all lanes are annotated with both 2D and 3D information in every frame. For the purpose of this project, we will only be using the 2D lane annotations.

To prepare the dataset for training, you may follow the steps below.

### 1. Download OpenLane

Follow the [download instructions](https://github.com/OpenDriveLab/OpenLane/tree/main#download) given by OpenLane.

The only folders needed are all the image folders and `lane3d_300`, which is a subset of the dataset that contains annotations for 300 sequences of lanes. Note that you can also use `lane3d_1000`, which would give the whole dataset. 

From now, we will refer to `lane3d_300`/`lane3d_1000` as `annotations` for greater clarity.

### 2. Organise folder structure

Organise the folder structure as follows:
```
├── images        #to be known as IMAGE_DIR
|   ├── training
|   |   ├── segment-xxx
|   |   |   ├── xxx.jpg
|   |   |   └── ...
|   └── validation
|       ├── segment-xxx
|       |   ├── xxx.jpg
|       |   └── ...
├── annotations    #to be known as ANN_DIR
|   ├── training
|   |   ├── segment-xxx
|   |   |   ├── xxx.json
|   |   |   └── ...
|   ├── validation
|   |   ├── segment-xxx
|   |   |   ├── xxx.json
|   |   |   └── ...
|   └── test
|       ├── curve_case
|       |   ├── segment-xxx
|       |   |   ├── xxx.json
|       |   |   └── ...
|       ├── ...
|       ├── curve_case.txt
|       ├── xxx.txt

```

This is important to ensure that the folder structure is compatible with our preprocessing code.

### 3. Preprocess dataset
OpenPifPaf is built with a default dataloader that works with COCO-style annotations. To make use of this, instead of writing our own dataloader, we have decided to transform the OpenLane annotations into a single COCO-style `.json` file instead. We have written a script, `openlane_to_coco.py` for this reason. 

`openlane_to_coco.py` will take all the 2D lane annotations, downsample them to 24 points, calculate a bounding box that contains all 24 points, and combine the information into a single `.json` file following the annotation style of the COCO dataset. You may read [this](https://opencv.org/introduction-to-the-coco-dataset/) to find out more about COCO dataset.

To use the script, simply replace the file paths as needed and run the following command:

```bash
python3 -m openpifpaf_openlane.openlane_to_coco \
    --dir_data= ANN_DIR \
    --dir_images= IMAGE_DIR \
    --dir_out= OUTPUT_DIR \
    --sample
```

This will produce two big `.json` files, one containing annotations for training and one for validation.
As the original number of annotations is huge and will require a lot of computation for training, we have chosen to keep just 10% of the annotations by providing the `--sample` argument. You may choose to omit this argument if you have the computational means. 

### 4. Visualise processed data
We have provided a Jupyter notebook, `visualise_annotations.ipynb`, that you can use to visualise the annotations on top of the original images. This is a sample of how a COCO-style annotation is supposed to look like on OpenLane data:

<p align="center" width="100%">
    <img width="50%" src="https://github.com/tlyi/CIVIL-459-Project/assets/69505852/c3e87524-bd2b-4870-b0bd-596e2e6d9d5b">
</p>

To compare it with the original annotations by OpenLane, you may use `visualise_annotations_openlane.ipynb`.


## Code

### Train 

For training on OpenLane dataset, you can run `./train_job.sh` after proper modifications to the `train.bat` parameters and directories. Note that our bash script is written with the intention to submit a batch job to HPC resources like SCITAS. If you do not require this, do alter the commands as you deem fit. 

You may also simply run the following code on the command line, with modification to parameters you would like to tune and experiment with.

```
python3 -m openpifpaf.train --lr=0.002 --momentum=0.9 --b-scale=5.0 \
  --epochs=1000 \
  --lr-warm-up-factor=0.25 \
  --output=outputs/openlane_train \
  --batch-size=5  --val-batches=1 --val-interval=10 \
  --weight-decay=1e-5 \
  --dataset=openlane \
  --basenet=shufflenetv2k16 \
  --openlane-train-annotations <dir of training annotations>
  --openlane-val-annotations <dir of validation annotations> \
  --openlane-train-image-dir <dir of training images> \
  --openlane-val-image-dir <dir of validation images>\
  --loader-workers 1
```
The `--basenet` argument specifies which base network the model should start training on. It is recommended to use a base network so that the model does not have to learn everything from scratch. OpenPifPaf comes with [several options for basenet](https://openpifpaf.github.io/cli_help.html#train) and we have chosen to use ShuffleNet. 

Some other useful flag arguments include:

```
--debug                             # print debug messages (default: False)
                                    # and turn off the dataset shuffle 
                            
--checkpoint CHECKPOINT             # train on top of previously trained
                                    # checkpoints, this cannot be used 
                                    # with --basenet at the same time.

--train-batches TRAIN_BATCHES
                                    # specify the number of batches 
                                    # trained per epoch, you can 
                                    # assign it to a small number 
                                    # like 1 to overfit on single image 

--openlane-no-augmentation
                                    # do not apply data augmentation on OpenLane
                             
```

### Predict

Prediction runs using the standard openpifpaf predict command, but using the model we have trained specifically on lanes. The command line argument is as follows:

 ```
 python3 -m openpifpaf.predict  \
         --checkpoint <specify the checkpoint path> \ 
         --force-complete-pose --debug-indices cif:0 caf:0 \
         --long-edge=425 --loader-workers=1 \ 
         --save-all # save every debug plot
         --image-output <optionally specify output dir and file name> \
         <path of the image you want to perform prediction on>  
```
You can also decide whether to output a json file, with the option to specify the output path or directory (default: None) using `--json-output`.
In addition, you can also decide whether you want to interactively show the plots in matplotlib with `--show` (use itermplot, for instance)
or to save image files with `--save-all` (defaults to the all-images/ directory). 

The necessary scripts are already in `predict.sh`, you can simply run `./predict.sh` after modifications.


### Evaluate

To evaluate the pretrained model, use:
```
python3 -m openpifpaf.eval \
--dataset=openlane --loader-workers=1 \
--checkpoint <checkpoint path> \
--force-complete-pose --seed-threshold=0.2 \ 
```
Evaluation metrics like average precision (AP) are created with this tool.

## Experimental Setup

To ensure that we get desirable results, we carried out a few rounds of trial and experimentation. Since there is no known example of OpenPifPaf being used on lanes, our strategy was to just try, observe the results and improve from there. 

#### 1. Modelling the lanes as 24 keypoints
In the original annotations, the lanes are identified by a varying number of keypoints, ranging from tens to hundreds. However, OpenPifPaf requires there to be a fixed number of keypoints so that the network is able to learn how to identify and connect them together. We chose 24 keypoints as it has been used to [identify car poses](https://openpifpaf.github.io/dev/plugins_apollocar3d.html#evaluation) successfully, so we know that it is not too few or many for the network. It also seemed to be a sufficient number to capture details in lane poses, such as turns. 

To provide the network with a skeleton to work with, we simply plot out 24 keypoints and connected them to each other. The keypoints were obtained by uniformly downsampling from the original lane annotations.

#### 2. Modelling just the start of the lane (1 keypoint)
We realised that using our initial downsampling method, we are asking the model to detect **exactly 24** points on the lane mark, which is not fair for the model. This is because unlike human joint keypoints, which are learnable and localisable with visual cues, the model is unable to differentiate between different keypoints in the middle of the lane, and there is no way to tell it which point is better than the ones next to it. Hence, we decided to simplify the task by keeping only the closest keypoint, essentially asking the model to detect just the start of the lane. 
Using this method, we observed that the results were more consistent. 

#### 3. Modelling the start and end of the lane (2 keypoints)

While we were able to yield consistent result using only one keypoint, this information is essentially not meaningful in the lane detection task, Therefore, we also implemented our plugin using only 2 keypoints with a new downsample strategy: keeping only the closest and the furthest lane points. Ideally, the model would learn to detect the start and end points of the lane and link them together. 

This is certainly just a coarse straight-line estimation for the lane and it cannot model more complex lanes such as turns. However, it paves the way for the subsequent development of a more reasonable downsample strategy for middle points.


If you are also interested in verifying our 2-kps model, you can simply first, delete the two files named `constants.py` and `openlane_kp.py`, and then, change the name of `constants_2kps.py` to `constants.py`, `openlane_kp_2kps.py` to `openlane_kp.py`. For trainer reading in annotations smoothly, you will need to redo the preprocess of data using `openlane_to_coco_2kp.py` as described in [Dataset Description](#dataset-description). This will extract only the start and end point coordinates annotation of every lane. Do not forget to specify the updated json file path in your trainning code.

#### 4. Overfitting on a single image
For all the above mentioned methods, to verify that our methods are working, before training on the big dataset formally, we first performed overfitting on a single image for 1000 epochs, with a learning rate of 0.001. 

These are the results after overfitting on the 2-keypoints model.
<p align="center" width="100%">
    <img width="50%" src="https://github.com/tlyi/CIVIL-459-Project/assets/69505852/ed88bd36-d573-4faa-97d3-6a9a3b65785c">
</p>


As observed, the model seems quite able to visually detect every lane. However, it seems to connect points from different lanes when there's visual occlusion from other vehicles. This is to be expected as 2 keypoints is simply too little information for the model to make accurate dedictions when there is loss of information.


#### 5. Experiment with learning rates
After several rounds of trials and error, we settled on a learning rate of 0.001. While we observed that the loss was decreasing steadily over time, we found that the rate of decrease was rather slow, which we deduced could be attributed to a low learning rate. However, when we increased the learning rate to 0.002, we experienced infinite loss. Hence, we decided to keep the learning rate at 0.001.

## Results

The below images visualises the components of Composite Intensity Field (CIF) and Composite Association Field (CAF) for the closest keypoint and finally outputs the overall prediction. CIF characterises the intensity of predicted keypoints and CAF characterises the intensity of predicted association between keypoints. Together, these two components enable the model to identify and form connections between keypoints. For more information about CIF and CAF, you may refer to the [paper](https://arxiv.org/abs/2103.02440) written by the creators of OpenPifPaf. 

*   **24 keypoints**
<p align="center" width="100%">
    <img width="50%" src="https://github.com/tlyi/CIVIL-459-Project/assets/118620053/9f4784e7-af24-4621-ad0f-65617e4d1fb8">
</p>

*   **2 keypoints (closest and furthest)**
<p align="center" width="100%">
    <img width="50%" src="https://github.com/tlyi/CIVIL-459-Project/assets/118620053/702cb3b5-97fc-4dc0-83eb-2f8716ef9693">
</p>



The below images show the comparison between predictions on validaton images using 24 keypoints (top) and 2 keypoints (bottom). These images were generated with the flag `--force-complete-pose` enabled as the models were unable to connect the keypoints well without it. While the outline of the lanes were modelled well, the model predicts it with very low confidence. We deduce that this could either be attributed to insufficient training epochs, or an inappropriately defined loss function (elaborated further [here](#2-redefine-evaluation-metrics)). 

<p align="center" width="100%">
    <img width="100%" src="https://github.com/tlyi/CIVIL-459-Project/assets/69505852/35edc49b-4afe-48e3-af65-32740b9c46c7">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/tlyi/CIVIL-459-Project/assets/69505852/fb5df390-e021-4737-a55a-6fce8f0c302c">
</p>

Using both methods, straight lanes are properly detected. However, as expected, curved lanes are simplified using just 2 keypoints, while with 24 keypoints, the curves are captured quite well. 


## Further Improvements
#### 1. Redefine loss function

The loss function is defined by [Kreiss et al.](https://arxiv.org/abs/2103.02440) and it consists of confidence, localization and scale. We observed that for the 24 keypoints model, the loss at the end of 30 epochs was still very high, at around 5000, and for the 2 keypoints, after 80 epochs, it was at ~75. Even though the loss was decreasing progressively with each epoch, it was still very high in the end. This could potentially be the reason why the model is unable to predict the lanes with high confidence. We suspect that the loss function may not be directly applicable on our task of lane detection. As [previously elaborated on](#2-modelling-just-the-start-of-the-lane-1-keypoint), it is unfair to expect the model to pinpoint keypoints on the lane the same way it detects distinct keypoints on human poses. More research could be donem focusing on published papers that deal with lane detection specifically, to come up with a more appropriate loss function.

#### 2. Redefine evaluation metrics or improve downsampling strategy
The metrics used for evaluation follows [COCO's](https://arxiv.org/abs/2103.02440) keypoint evaluation method. The object keypoint similarity [(OKS)](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) score is used to assign a bounding box to each keypoint as a function of the person instance bounding box area. The metric computes overlaps between ground truth and predicted bounding boxes for each keypoints to compute the standard detection metrics average precision (AP) and average recall (AR). As explored before, this is reasonable for human joints, but there is possibly no overlap at all between bounding boxes of 2 reasonable points near each other on the same lane. 

We can either investigate the feasible scale and position of bounding box assigning for a lane keypoint to enlarge overlapping between 2 neighbor points on the lane, or implement starting point,distance rule, and ending point during keypoint downsampling for training to make the detection learnable.    

#### 3. Extend to 3D space
Since the COCO annotations were designed to work with 2D objects, we found it difficult to translate it to 3D space. Specifically, we were unsure how we could define a bounding box in 3D space. To overcome this, it is possible to instead write our own dataloader, or combine OpenPifPaf with state-of-the-art spatial transformation methods like [PersFormer](https://arxiv.org/abs/2203.11089).

## Conclusion
We have visually demonstrated the feasibility of extending the capabilities of OpenPifPaf to the task of lane detection.
While there exists plenty of space for improvement, we are satisfied with the results we have achieved given the time and resource constraints, and believe that it still serves as a meaningful preliminary proof of concept for the task. 

Through working on this project, we were able to apply all our learnings from this course and see for ourselves the massive potential that deep learning has to offer in the field of autonomous vehicles. It has truly been a fruitful journey. We are grateful for the help from teaching team.

## References
Kreiss, S., Bertoni, L., &amp; Alahi, A. (2022). OpenPifPaf: Composite fields for semantic keypoint detection and spatio-temporal association. IEEE Transactions on Intelligent Transportation Systems, 23(8), 13498–13511. https://doi.org/10.1109/tits.2021.3124981 

Chen, L., Sima, C., Li, Y., Zheng, Z., Xu, J., Geng, X., Li, H., He, C., Shi, J., Qiao, Y., &amp; Yan, J. (2022). PERSFORMER: 3D lane detection via&nbsp;perspective transformer and&nbsp;the&nbsp;openlane benchmark. Lecture Notes in Computer Science, 550–567. https://doi.org/10.1007/978-3-031-19839-7_32 


