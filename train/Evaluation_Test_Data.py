# importing the required modules
import os
import subprocess
import json
import random
import cv2
import json
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine.defaults import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config, PanopticDeeplabDatasetMapper # noqa
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.evaluation import COCOPanopticEvaluator, COCOEvaluator, DatasetEvaluators

Visualization = False

categories_file_path = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/Categories.json"
with open(categories_file_path, "r") as json_file:
    categories_data = json.load(json_file)

stuff_names = [cat["name"] for cat in categories_data]
thing_names = [cat["name"] for cat in categories_data]

stuff_ids = [cat["id"] for cat in categories_data if cat["isthing"]==0]
thing_ids = [cat["id"] for cat in categories_data if cat["isthing"]==1]

total_classes = len(stuff_ids) + len(thing_ids)

# stuff_contigous_ids = { id:i for i, id in enumerate(stuff_ids)}
# thing_contigous_ids = { id:i for i, id in enumerate(thing_ids)}


stuff_contigous_ids = dict((zip(stuff_ids, list(range(0, len(stuff_ids))))))
thing_contigous_ids = dict(zip(thing_ids,list(range(len(stuff_ids), total_classes))))

# thing_contigous_ids = dict(zip(thing_ids,list(range(0, len(thing_ids)))))
# stuff_contigous_ids = dict(zip(stuff_ids, list(range(len(thing_ids), total_classes))))

print("+"*50)

print("stuff_classes:", stuff_names)
print("stuff_ids:", stuff_ids)
print("thing_classes:", thing_names)
print("thing_ids", thing_ids)


metadata_dict = { "stuff_classes": stuff_names,
                "thing_classes": thing_names,
                "thing_dataset_id_to_contiguous_id": thing_contigous_ids,
                "stuff_dataset_id_to_contiguous_id": stuff_contigous_ids}

print("+"*50)
print(metadata_dict)


data_name = 'Custom_ZED_CAM_pedestrian_data_'

data_root_dir = '/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset'

for i in ["train", "test"]:

    root_dir = os.path.join(data_root_dir, i)
    assert os.path.isdir(root_dir), "Directory does not exist"
    image_root = os.path.join(root_dir, i + "_images")
    panoptic_root = os.path.join(root_dir, i + "_coco_panoptic")
    panoptic_json = os.path.join(root_dir, i + "_coco_panoptic.json")
    with open(panoptic_json) as panoptic_json_file:
        panoptic_dict = json.load(panoptic_json_file)
    sem_seg_root = os.path.join(root_dir, i + "_semantic_segmentation_masks")
    instances_json = os.path.join(root_dir, i + "_coco_panoptic_instance.json")
    register_coco_panoptic(data_name + i, metadata_dict, image_root, panoptic_root, panoptic_json,
                        instances_json)
    dataset_dicts = DatasetCatalog.get(data_name + i)
    # print("sample loaded dataset dict:", dataset_dicts[0])

train_metadata = MetadataCatalog.get(data_name + "train")


if Visualization:

    for i in ["train", "test"]:
        dataset_dicts = DatasetCatalog.get(data_name + i )
        for d in random.sample(dataset_dicts, 5):
            print(f'visualizing {i} dataset' )
            img = cv2.imread(d["file_name"])
            print(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata= train_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("Image", out.get_image()[:, :, ::-1])

            # Wait for a key press
            cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()

cfg = get_cfg()

config_file = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_9/config.yaml"

add_panoptic_deeplab_config(cfg)
cfg.merge_from_file(config_file)

cfg.DATASETS.TRAIN = (data_name + "train",)
cfg.DATASETS.TEST = (data_name + "test",)
# cfg.TEST.EVAL_PERIOD = 100
# cfg.DATALOADER.NUM_WORKERS = 1

# cfg.MODEL.DEVICE='cpu'
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_9/model_0002399.pth"
# cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = total_classes-1
# cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = 256 #512 #128
# cfg.MODEL.BACKBONE.FREEZE_AT = 3

# cfg.SOLVER.IMS_PER_BATCH = 6
# cfg.SOLVER.BASE_LR = 0.0001
# cfg.SOLVER.MAX_ITER = 3000


# cfg.INPUT.CROP.ENABLED = False
# cfg.INPUT.MASK_FORMAT = "bitmask"

predictor = DefaultPredictor(cfg)


Folder = True
Single_image = False

if Folder:
        
    # Save the image to a folder
    test_image_folder = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/test/test_images"
    #"/home/yalamaku/Documents/Thesis/Dataset_files/TUK_uni_dataset/sample_selected_test_images"
    #"/home/yalamaku/Documents/Thesis/Dataset_files/Unseen_Images_testing"
    #"/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/test/test_images"
    output_folder = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Test_Data_Infer_Results_2/train_9/TestData"  # Specify the desired output folder
    test_image_list = os.listdir(test_image_folder)
    for image in test_image_list:
        if image.endswith('.jpg') or image.endswith('.png'):
            # for single test image
            im = cv2.imread(os.path.join(test_image_folder, image))
            panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
            v = Visualizer(im[:, :, ::-1], train_metadata, scale=1.2)
            v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

            output_path = os.path.join(output_folder, os.path.basename(image))
            cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
            print(f'output saved to path: {output_path}')

if Single_image:

    image_path = "/home/yalamaku/Documents/Detectron_2/Dataset/TUK_uni_dataset/images/frame_0293.jpg"
    # for single test image
    im = cv2.imread(image_path)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], train_metadata, scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

    output_path = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Test_Data_Infer_Results/Train_8/New_Data/img_0293.png"
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
