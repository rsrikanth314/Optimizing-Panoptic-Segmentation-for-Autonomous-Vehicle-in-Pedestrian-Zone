import onnxruntime as ort
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
import os
import json
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic

import time

#import pdb
# pdb.set_trace()

# onnx model path
onnx_model_path = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/export_model_onnx_test/model_train_weight_2_train_1_0001799_model.onnx"


quantized_model_path = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/export_model_onnx_test/model_train_weight_2_train_1_0001799_quantized_model.onnx"
# # Quantizing the model
quantized_model = quantize_dynamic(onnx_model_path, quantized_model_path)

# Avialable execution providers
print(ort.get_available_providers())

# Explicitly specify the execution providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # You can specify other providers if needed 'CUDAExecutionProvider', 'CPUExecutionProvider'

# Create the InferenceSession with the specified providers
ort_session = ort.InferenceSession(quantized_model_path, providers=providers)

# Load the image and convert it to RGB formate
image_path = "/home/yalamaku/Documents/Thesis/Onnx_Models/Onnx_finroc_test_images/34.jpg"
org_input_image = cv2.imread(image_path)[:,:,::-1]

# inference speed start timer
start_time = time.time()

# onnx model expected inputs and output shapes
Model_input_height, Model_input_width = (1024, 1820)  # onnx input shape
Model_output_height, Model_output_width = (376, 672)  # onnx output shape

# Resize the image to the expected input size of the model
model_input_image = cv2.resize(org_input_image.copy(), (Model_input_width, Model_input_height)).astype(np.float32) #(np.float32) #(np.uint8)

# Transpose the image to match the model's input format if necessary

model_input_image = np.transpose(model_input_image, (2, 0, 1))  # Channels-first format (C, H, W)

print(f" orginal input image shape: {org_input_image.shape}")
print(f"Model input_image shape: {model_input_image.shape}")

# Onnx input schema
ort_inputs = {ort_session.get_inputs()[0].name: model_input_image}

# Run inference
ort_outputs = ort_session.run(None, ort_inputs)

# Always check the output before displaying
print(f" Output image shape: {ort_outputs[0].shape}")
print(f" output type: {type(ort_outputs)}")
print(len(ort_outputs))
print(type(ort_outputs[0]))
print(ort_outputs[0].shape)
print(type(ort_outputs[0]))
print(ort_outputs[0].shape)

# Save the panoptic seg raw model output
Onnx_Raw_output_path = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/ONNX_runtime_images/Model_weights_2_train_1_0001799_raw_output.png"
# Convert the output image data to uint8 and rescale it to [0, 255]
Model_raw_output = (ort_outputs[0] * 255).astype(np.uint8)
cv2.imwrite(Onnx_Raw_output_path, Model_raw_output)

end_time = time.time()

inference_time = end_time - start_time
print(f"Total Inference Time: {inference_time} seconds")

fps = 1 / inference_time

print(f"Frames Per Second (FPS): {fps}")


# # Display the output image
# cv2.imshow("image", Model_raw_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Register Custom Dataset 
Visualization = False

categories_file_path = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/Categories.json"
with open(categories_file_path, "r") as json_file:
    categories_data = json.load(json_file)

stuff_names = [cat["name"] for cat in categories_data]
thing_names = [cat["name"] for cat in categories_data]

stuff_ids = [cat["id"] for cat in categories_data if cat["isthing"]==0]
thing_ids = [cat["id"] for cat in categories_data if cat["isthing"]==1]

total_classes = len(stuff_ids) + len(thing_ids)


stuff_contigous_ids = dict((zip(stuff_ids, list(range(0, len(stuff_ids))))))
thing_contigous_ids = dict(zip(thing_ids,list(range(len(stuff_ids), total_classes))))

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

# train_metadata = MetadataCatalog.get("cityscapes_fine_panoptic_train")

panoptic_seg = torch.from_numpy(ort_outputs[0]).to(torch.int64)

v = Visualizer(cv2.resize(org_input_image, (Model_output_width, Model_output_height)), train_metadata, scale=1.2)
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info= None)

output_path = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/ONNX_runtime_images/Model_weights_2_train_1_0001799_color_seg_output.png"
cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
