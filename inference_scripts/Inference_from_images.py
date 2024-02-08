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


def setup_config():
    # Load the Detectron2 configuration file
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml")  # Replace with your config file path
    cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Detectron_2/downloaded_weights/COCO_dsconv.pkl" 
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.CROP.ENABLED = False

    return cfg


def perform_panoptic_segmentation(cfg, image_path):
    # Create the predictor
    predictor = DefaultPredictor(cfg)

    # Load the image
    image = cv2.imread(image_path)

    # Perform panoptic segmentation
    outputs = predictor(image)

    return outputs

def save_segmentation_result(outputs, output_folder, image_name):
    panoptic_seg = outputs["panoptic_seg"].numpy()
    output_path = os.path.join(output_folder, f"{image_name}_segmentation.png")
    cv2.imwrite(output_path, panoptic_seg)



if __name__ == "__main__":
    images_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_Cam_Extracted_data/images/"  # Replace with the path to your images folder
    output_folder = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_Cam_Extracted_data/infer_images_output"  # Replace with the path where you want to save the output images

    os.makedirs(output_folder, exist_ok=True)

    cfg = setup_config()

    image = 0

    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        outputs = perform_panoptic_segmentation(cfg, image_path)
        save_segmentation_result(outputs, output_folder, os.path.splitext(image_name)[0])
        print("image : ", image = image + 1)
