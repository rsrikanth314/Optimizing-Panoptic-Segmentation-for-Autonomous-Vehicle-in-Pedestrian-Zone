import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)


import random
import json
import cv2
from detectron2.data.datasets import register_coco_panoptic
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def setup_cfg():

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

    train_metadata = MetadataCatalog.get(data_name + "train")


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

    config_file = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights/Train_7/config.yaml"
    
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    #add_pointrend_config(cfg)
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights/Train_7/model_0001999.pth"

    # cfg.DATASETS.TRAIN = ('Custom_ZED_CAM_pedestrian_data_train',)
    # cfg.DATASETS.TEST = ('Custom_ZED_CAM_pedestrian_data_valid',)

    # cfg.TEST.EVAL_PERIOD = 100
    # cfg.MODEL.DEVICE='cuda'    #('cuda', 'cuda:0', 'cpu)

    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10
    # cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.20
    # cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.20
    # #cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = 256 #512 #128
    # cfg.MODEL.BACKBONE.FREEZE_AT = 3

    # cfg.DATALOADER.NUM_WORKERS = 1

    # cfg.SOLVER.IMS_PER_BATCH = 1 #3
    # cfg.SOLVER.BASE_LR = 0.0001
    # cfg.SOLVER.MAX_ITER = 2500
    # cfg.SOLVER.CHECKPOINT_PERIOD = 100
    # cfg.SOLVER.OPTIMIZER = "ADAM"
    # cfg.SOLVER.WEIGHT_DECAY =0.001
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.001
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.001

    # # cfg.INPUT.MIN_SIZE_TRAIN = (256, 320, 384, 448, 512)
    # # cfg.INPUT.MIN_SIZE_TRAIIN_SAMPLING = "choice"
    # # cfg.INPUT.MIN_SIZE_TEST = 376
    # # cfg.INPUT.MAX_SIZE_TRAIN = 2016
    # # cfg.INPUT.MAX_SIZE_TEST = 672

    cfg.INPUT.CROP.ENABLED = False
    # # cfg.INPUT.CROP.SIZE = (256, 512)
    # cfg.INPUT.MASK_FORMAT = "bitmask"

    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs, format, output):
    from detectron2.export import Caffe2Tracer

    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(output, "model.svg"), inputs=inputs)
        return caffe2_model
    elif format == "onnx":
        import onnx

        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(output, "model.onnx"))
    elif format == "torchscript":
        ts_model = tracer.export_torchscript()
        with PathManager.open(os.path.join(output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, output)


# experimental. API not yet final
def export_scripting(torch_model, format, output):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    assert format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs, format, output):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, output)
    elif format == "onnx":
        with PathManager.open(os.path.join(output, "model.onnx"), "wb") as f:
              torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION, verbose= False) #operator_export_type= torch.onnx.OperatorExportTypes.ONNX_ATEN)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(sample_image):

    if sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


if __name__ == "__main__":

    export_method = "tracing"  # (caffe2_tracing, tracing, scripting)
    format = "onnx"             # (caffe2, onnx, torchscript)
    run_eval = False    # (True, False)

    sample_image = "/home/yalamaku/Documents/Thesis/Dataset_files/TUK_uni_dataset/images/frame_1764.jpg"
    #"/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/test/test_images/12.jpg"
    
    output_folder = "/home/yalamaku/Documents/Thesis/my_scripts/ONNX_export_scripts/export_model_onnx_test"
    logger = setup_logger()
    PathManager.mkdirs(output_folder)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg()

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # convert and save model
    if export_method == "caffe2_tracing":
        sample_inputs = get_sample_inputs(sample_image)
        exported_model = export_caffe2_tracing(cfg, torch_model, sample_inputs, format, output_folder)
    elif export_method == "scripting":
        exported_model = export_scripting(torch_model, format, output_folder)
    elif export_method == "tracing":
        sample_inputs = get_sample_inputs(sample_image)
        exported_model = export_tracing(torch_model, sample_inputs, format, output_folder)

    # run evaluation with the converted model
    if run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={export_method}, format={format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, output_dir=output_folder)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")
