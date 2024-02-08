from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import PanopticDeeplabDatasetMapper, add_panoptic_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils
import detectron2.data.transforms as T
import torch
from detectron2.export import TracingAdapter
import random
import json
import cv2
import os
from detectron2.data.datasets import register_coco_panoptic
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import onnx

def main():

    # register custom Dataset 
    Visualization = False

    categories_file_path = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/Categories.json"
    with open(categories_file_path, "r") as json_file:
        categories_data = json.load(json_file)

    stuff_names = [cat["name"] for cat in categories_data]
    thing_names = [cat["name"] for cat in categories_data]

    stuff_ids = [cat["id"] for cat in categories_data if cat["isthing"]==0]
    thing_ids = [cat["id"] for cat in categories_data if cat["isthing"]==1]

    total_classes = len(stuff_ids) + len(thing_ids)

    print(f" total number of classes:{total_classes}")

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

    for i in ["train", "valid"]:

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

    if Visualization:

        for i in ["train", "valid"]:
            dataset_dicts = DatasetCatalog.get(data_name + i )
            for d in random.sample(dataset_dicts, 5):
                print(f'visualizing {i} dataset' )
                img = cv2.imread(d["file_name"])
                print(d["file_name"])
                visualizer = Visualizer(img[:, :, ::-1], metadata= MetadataCatalog.get(data_name + i), scale=0.5)
                out = visualizer.draw_dataset_dict(d)
                cv2.imshow("Image", out.get_image()[:, :, ::-1])

                # Wait for a key press
                cv2.waitKey(0)

            # Close all windows
            cv2.destroyAllWindows()

    # train_metadata = MetadataCatalog.get(data_name + "train")

    config_file = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_1/config.yaml"
    
    #set up config
    cfg = get_cfg()
    
    # Add Panoptic Deeplab config
    add_panoptic_deeplab_config(cfg)

    # Add sepcific config from file
    cfg.merge_from_file(config_file)

    # TODO: if nescessary, update the configs here
    cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_1/model_0001799.pth"
    # cfg.INPUT.CROP.ENABLED = False

    # Make config immutable after this point
    cfg.freeze()

    # create a torch model and load checkpoint
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # sample data
    sample_image = "/home/yalamaku/Documents/Thesis/Dataset_files/TUK_uni_dataset/sample_selected_test_images/frame_1825.jpg"
    
    original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
    
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    # input = {"image": image, "height": height, "width": width}
    inputs = [{"image": image}]

    onnx_model_path_traced = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/export_model_onnx_test/model_train_weight_2_train_1_0001799_model_2.onnx"

    traceable_model = TracingAdapter(torch_model, inputs, inference_func= None)

    #Export to onnx model
    dummy_input_names = ["input"]
    dummy_output_names = ["Panoptic_image"]
    dynamic_axes = {
        "input": {0: "channel", 1: "height", 2: "width"},
        "Panoptic_seg": {0: "height", 1: "width"},
    }

    # Uncomment this and the post proceeesng code in panoptic_seg.py if you want to output all the data below
    #dummy_output_names = ["instance_center_points", "instance_ids", "instance_mask_bool", "instance_points", "output_dimension", "Panoptic_seg", "Segments_info"]
    # Define dynamic axes for input and output
    # dynamic_axes = {
    #     "input": {0: "channel", 1: "height", 2: "width"},
    #     "instance_center_points": {0: "instance", 1: "center_points"},
    #     "instance_ids": {0: "instance"},
    #     "instance_mask_bool": {0: "instance", 1: "height", 2: "width"},
    #     "instance_points": {0: "instance"},
    #     "output_dimension": {0: "height", 1: "width"},
    #     "Panoptic_seg": {0: "height", 1: "width"},
    #     "Segments_info": {0: "total_class_preds", 1: "height", 2: "width"}
    # }

    torch.onnx.export(model= traceable_model,
                      args= (image,),
                       f= onnx_model_path_traced,
                        export_params= True,
                         verbose= False,
                          operator_export_type= torch.onnx.OperatorExportTypes.ONNX,
                          opset_version= 15, 
                          input_names=dummy_input_names, 
                          dynamic_axes=dynamic_axes, 
                          output_names= dummy_output_names
    )
    
    # import pdb
    # pdb.set_trace()
    # Dynamo export 

    # export_options = torch.onnx.ExportOptions(dynamic_shapes= True)
    # export_output = torch.onnx.dynamo_export(torch_model, inputs, export_options= export_options)
    # export_output.save('/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/export_model_onnx_test/dynamo_export.onnx')


    # # Discover all unconvertible ATen Ops at once
    # torch_script_graph, uncovertible_ops = torch.onnx.utils.unconvertible_ops(traceable_model, (image,), opset_version= 11)
    # print(set(uncovertible_ops))  # Results : {'aten::mode', 'aten::__and__'}

    # Load and check the exported onnx model
    onnx_model = onnx.load(onnx_model_path_traced)
    onnx.checker.check_model(onnx_model)
    
if __name__ == "__main__":

    main()