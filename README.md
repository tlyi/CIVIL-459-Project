# OpenPifPaf Lane Detection

## Quick Navigation
--------------------
- [Project Description](#project-description)
- [Contribution Overview](#contribution-overview)
- [Installation](#installation)


## Project Description
--------------------
This project is part of EPFL's "Deep Learning for Autonomous Vehicles" course. 
This year, the final goal of the course is to build the main computer vision components of an autonomous vehicle. This project aims to contribute to one small but important module of this big system, 3D lane detection. 

We are inspired by the idea of [OpenPifPaf](https://openpifpaf.github.io/intro.html), which achieves human pose-estimation by detecting and associating spatial-temporal human joint keypoints.

We see the potential in using the same concept to simplify the task of detecting and regressing potentially hundreds or even thousands of pixel points of road lane to just a few key points and connect them to form an estimate of a lane. We hope that this opens up hope for a new framework that can significantly reduce the time required for lane detection.

## Contribution Overview
--------------------

In summary, our contribution is to extend the function of [OpenPifPaf](https://openpifpaf.github.io/intro.html) to lane detection by enabling it to be trained on a whole different dataset: [OpenLane](https://github.com/OpenDriveLab/OpenLane). We transformed OpenLane dataset to CoCo format and downsampled originallane annotations to several keypoints (24 and 2, respectively). Plugin necessities were implemented without changing the main body of openpifpaf, making the project easdy to install and set up. To note that due to the very different nature of the two datasets plus time and resource limit, our progress is currently mainly on 2d lane detection with massive debugging, explorations, trial and error, but the preliminary results show the feasibility of this idea and clear way to 3d lane detection extension.

## Installation
-------------
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
The dataset that we are using is very big and will take days to train. We have provided a [checkpoint](https://drive.google.com/file/d/1IEKkXFKS5HWgyEEhrRoDiCdvLgZRtmV7/view?usp=sharing) that has already been trained on 30 epochs. You may choose to train either from scratch (not recommended), or from one of the backbones provided by OpenPifPaf, or on top of our provided checkpoint. 

## Dataset Description
--------------
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


To compare it with the original annotations by OpenLane, you may use `visualise_annotations_openlane.ipynb`.


## Code
------
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

## Experimental setup
------------------
To ensure that we get desirable results, we carried out a few rounds of trial and experimentation. Since there is no known example of OpenPifPaf being used on lanes, our strategy was to just try, observe the results and improve from there. 

#### 1. Modelling the lanes as 24 keypoints
In the original annotations, the lanes have varying number of keypoints, ranging from tens to hundreds. However, OpenPifPaf requires there to be a fixed number of keypoints so that the network is able to learn how to identify and connect them together. We chose 24 keypoints as it has been used to [identify car poses](https://openpifpaf.github.io/dev/plugins_apollocar3d.html#evaluation) successfully, so we know that it is not too few or many for the network. It also seemed to be a sufficient number to capture details in lane poses, such as turns. 

To provide the network with a skeleton to work with, we simply plot out 24 keypoints and connected them to each other. The keypoints were obtained by uniformly downsampling from the original lane annotations.

#### 2. Overfitting on a single image
We managed to perform overfitting on a single image for 1000 epochs and also ran on 10% of the whole dataset for 30 epochs.

#### 3. Experiment with learning rates

#### 4. Modelling just the start of the lane (1 keypoint)

#### 5. Modelling the start and end of the lane (2 keypoints)



The metrics used for evaluation follows [COCO's](https://arxiv.org/abs/2103.02440) keypoint evaluation method. The object keypoint similarity [(OKS)](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) score is used to assign a bounding box 
to each keypoint as a function of the person instance bounding box area. Similar to detection, the metric computes overlaps between ground truth and predicted bounding boxes to compute the standard detection metrics average precision (AP) and average recall (AR).

It didn't take us very long to realize it is not fair for lane keypoint detection using our downsample method. By this method, we are asking the model to detect exactly **these 24** points on the lane mark. 
Human joint keypoints are learnable with all the visual cues, which is not the case for lane points. 

Therefore, we also implemented our plugin using only 2 keypoints with a new downsample strategy: keeping only the closest and the furthest lane points. Ideally, the model would learn to detect the start and end points of the lane and link them together. 
This is certainly just a coarse straight-line estimation for the lane and it cannot fit a turn well, however, it paves the way for the subsequent development of a more reasonable downsample strategy for middle points.
  

For this 2-keypoints model, we also conducted both overfitting experiments with a single image for 1000 epochs and also 5% of the original dataset for 80 epochs. 

## Results

*   **24 keypoints**




*   **2 keypoints (closest and furthest)**
#### Overfitting
After training for 1000 epochs of single image

## Further Improvements

## Conclusion

Short one

