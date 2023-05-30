"""
Convert openlane json files to one single json file with COCO format
"""

import os
import time
import json
import argparse

import numpy as np

# Packages for data processing, crowd annotations and histograms
try:
    import matplotlib.pyplot as plt  # pylint: disable=import-error
except ModuleNotFoundError as err:
    if err.name != 'matplotlib':
        raise err
    plt = None

from .constants import LANE_KEYPOINTS_24, LANE_SKELETON_24, IMAGE_WIDTH, IMAGE_HEIGHT

def cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #TODO: Alter the OpenLane dataset to follow the saame hierarchy as shown on their GitHub page 
    # (https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md)
    parser.add_argument('--dir_data', default='../annotations',
                        help='dataset annotations directory')
    parser.add_argument('--dir_images', default='../images',
                        help='dataset images directory')
    parser.add_argument('--dir_out', default='./data_openlane',
                        help='where to save annotations and files')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to only process the first 50 images')
    parser.add_argument('--single_sample', action='store_true',
                        help='Whether to only process the first image')
    args = parser.parse_args()
    return args

class OpenLaneToCoco:

    # Prepare json format
    map_sk = LANE_SKELETON_24

    sample = False
    single_sample = False

    def __init__(self, dir_dataset, dir_images, dir_out):
        """
        :param dir_dataset: Original dataset directory containing json annotations
        :param dir_images: Original dataset directory containing images
        :param dir_out: Processed dataset directory
        """

        assert os.path.isdir(dir_dataset), 'dataset directory not found'
        self.dir_dataset = dir_dataset
        self.dir_images = dir_images
        self.dir_out_ann = os.path.join(dir_out, 'annotations')

        os.makedirs(self.dir_out_ann, exist_ok=True)
       
        self.json_file_24 = {}

        training_files = []
        training_dir = os.path.join(dir_dataset, "training")
        for dir, _, files in os.walk(training_dir):
            for file in files:
                relative_file = os.path.join(dir, file)
                training_files.append(relative_file)

        validation_files = []
        validation_dir = os.path.join(dir_dataset, "validation")
        for dir, _, files in os.walk(validation_dir):
            for file in files:
                relative_file = os.path.join(dir, file)
                validation_files.append(relative_file)

        # Load train val split
        self.splits = {
            "training": training_files,
            "validation": validation_files,
        }

    def process(self):
        """
        Iterate all json annotations, process into a single json file compatible with coco format
        """

        for phase, ann_paths in self.splits.items(): #Iterate through training and validation (phases) annotations 
            #keep count?
            lane_counter = 0

            #Initiate json file at each phase 
            self.initiate_json() #single JSON file containing all COCO information 

            #Optional arguments
            if self.sample:
                #keep 10% of the dataset, uniformly distributed
                ann_paths = ann_paths[::10]

            if self.single_sample:
                ann_paths = self.splits['train'][:1]

            #Iterate through json files and process into COCO style
            for ann_path in ann_paths:
                f = open(ann_path) #o
                openlane_data = json.load(f) 
                
                """
                Update image field in json file
                """
                relative_file_path = openlane_data['file_path']
                file_path = os.path.join(self.dir_images, relative_file_path)
                img_name = os.path.splitext(file_path)[0]   # Returns tuple (file_name, ext)
                #each image has a unique image_id and each image can have multiple annotations
                img_id = int(img_name.split("/")[-1])
                
                if not os.path.exists(file_path):
                    continue

                #Should not be necessary to open each image file to determine size; images in dataset seem to be standardised
                # img = Image.open(file_path)
                # img_width, img_height = img.size
                
                dict_ann = {
                    'coco_url': "unknown",
                    'file_name': img_name + '.jpg',
                    'id': img_id,
                    'license': 1,
                    'date_captured': "unknown",
                    'width': IMAGE_WIDTH,
                    'height': IMAGE_HEIGHT}
                self.json_file_24["images"].append(dict_ann)

        
                #extract keypoints, visibility, category, and load into COCO annotations field
                lane_lines = openlane_data['lane_lines']

                """
                Update annotation field in json file for each lane in image
                """    
                for lane in lane_lines:
                    category_id = lane['category']
                    kp_coords = np.array(lane['uv']) #take the image coords, not the camera coords

                    #note kp_coords format is [[u],[v]]
                    num_kp = len(kp_coords[0])
                    if num_kp < 24:
                        continue
                    
                    #downsample to 24 kps
                    kp_coords = kp_coords[:,::num_kp//24]
                    #make sure to keep only the first 24 keypoints [u,v,1] in kps
                    kp_coords = kp_coords[:,:24]
                    #update num_kp to the new number of keypoints, it should be 24
                    num_kp = int(len(kp_coords[0]))

                    #TODO: figure out why number of points in visibility != len(uv) but = len(xyz).
                    #For now, assume all points have visibility = 1
                    # kp_visibility = lane['visibility'] 
                   
                    kps = []
                    #keypoints need to be in [xi, yi, vi format]
                    for u, v in zip(kp_coords[0], kp_coords[1]):
                        kps.extend([u, v, 1]) #Note: visibility might not be correct
                
                    #define bounding box based on area derived from 2d coords     
                    box_tight = [np.min(kp_coords[0]), np.min(kp_coords[1]),
                                np.max(kp_coords[0]), np.max(kp_coords[1])]
                    w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
                    x_o = max(box_tight[0] - 0.1 * w, 0)
                    y_o = max(box_tight[1] - 0.1 * h, 0)
                    x_i = min(box_tight[0] + 1.1 * w, IMAGE_WIDTH)
                    y_i = min(box_tight[1] + 1.1 * h, IMAGE_HEIGHT)
                    box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)
    
                    coco_ann = {
                        'image_id': img_id,
                        'category_id': category_id,
                        'iscrowd': 0,
                        'id': lane_counter,
                        'area': box[2] * box[3],
                        'bbox': box,
                        'num_keypoints': num_kp,
                        'keypoints': kps,
                        'segmentation': []}

                    self.json_file_24["annotations"].append(coco_ann)
                    lane_counter += 1

        
            self.save_json_files(phase)
            print(f'\nPhase:{phase}')
            print(f'JSON files directory:  {self.dir_out_ann}')
    

    def save_json_files(self, phase):
        name = 'openlane_keypoints_'
        if self.sample:
            name = name + 'sample_10'
        elif self.single_sample:
            name = name + 'single_sample_'

        path_json = os.path.join(self.dir_out_ann, name + phase + '.json')
        with open(path_json, 'w') as outfile:
            json.dump(self.json_file_24, outfile)
       

    def initiate_json(self):
        """
        Initiate Json for training and val phase
        """
        
        lane_kps = LANE_KEYPOINTS_24
        
        self.json_file_24["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                  date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                             time.localtime()),
                                  description=("Conversion of openlane dataset into MS-COCO"
                                               " format with 2D keypoints"))
        self.json_file_24["categories"] = [dict(name='unknown',
                                         id=0,
                                         skeleton = LANE_SKELETON_24,
                                         supercategory='lane',
                                         keypoints=[]), 
                                         dict(name='white-dash',
                                         id=1,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='white-solid',
                                         id=2,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='double-white-dash',
                                         id=3,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='double-white-solid',
                                         id=4,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='double-white-solid',
                                         id=4,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='white-ldash-rsolid',
                                         id=5,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='white-lsolid-rdash',
                                         id=6,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='yellow-dash',
                                         id=7,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='yellow-solid',
                                         id=8,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='double-yellow-dash',
                                         id=9,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='double-yellow-solid',
                                         id=10,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='yellow-ldash-rsolid',
                                         id=11,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='yellow-lsolid-rdash',
                                         id=12,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='left-curbside',
                                         id=20,
                                         supercategory='lane',
                                         keypoints=[]),
                                         dict(name='right-curbside',
                                         id=21,
                                         supercategory='lane',
                                         keypoints=[])]
        self.json_file_24["images"] = []
        self.json_file_24["annotations"] = []


        
def main():
    args = cli()

    # configure
    OpenLaneToCoco.sample = args.sample
    OpenLaneToCoco.single_sample = args.single_sample

    apollo_coco = OpenLaneToCoco(args.dir_data, args.dir_images, args.dir_out)
    apollo_coco.process()

if __name__ == "__main__":
    main()

