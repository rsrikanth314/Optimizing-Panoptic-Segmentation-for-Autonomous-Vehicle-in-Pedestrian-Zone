import os
import cv2
import numpy as np
import torch
import time

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# To use demo for Panoptic-DeepLab, please uncomment the following two lines.
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa

#set up configuration file
cfg = get_cfg()

# To use demo for Panoptic-DeepLab, please uncomment the following two lines.
add_panoptic_deeplab_config(cfg)
cfg.merge_from_file("/home/yalamaku/Documents/Detectron_2/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml")
cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Detectron_2/downloaded_weights/cityscapes_no_dsconv.pkl"
cfg.INPUT.CROP.ENABLED = False

# Set up predictornvidi
predictor = DefaultPredictor(cfg)

# Set up dataset metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Set up video input
cap = cv2.VideoCapture('/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_video.avi')
output_path = 'ZED_infer_Cityscapes_no_dsconv_dlwghts_F_mc.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# Initialize time variables
total_time = 0
loop_start_time = time.time()


ret, frame = cap.read()

while(ret):

    #Panoptic segmentation visualizer
    # using visualizer to draw the preditctions
    start_time = time.time()
    predicitons, segmentInfo = predictor(frame)["panoptic_seg"]
    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time

    #show full segmentation image
    v = Visualizer(frame[:,:,::-1], metadata, scale= 1)

    out = v.draw_panoptic_seg_predictions(predicitons.to('cpu'), segmentInfo, area_threshold= .1)
    output.write(out.get_image()[:,:,::-1])

    # Calculate and display elapsed time for each loop
    loop_end_time = time.time()
    loop_elapsed_time = loop_end_time - loop_start_time
    current_frame += 1
    print(f"frame:{current_frame}/{total_frames}. Elapsed time for each frame: {loop_elapsed_time:.4f} seconds")
    loop_start_time = loop_end_time

    #capture the video frame by frame
    ret, frame = cap.read()

#after the loop release the cap object
cap.release()
output.release()

# destroy all windows
cv2.destroyAllWindows()