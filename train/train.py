"""
Custom Panoptic-DeepLab Training Script
"""
import os
from typing import Any
import torch
import json
import random
import cv2
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, launch
from detectron2.data.datasets import register_coco_panoptic
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import (COCOPanopticEvaluator,
                                   COCOEvaluator,
                                   DatasetEvaluators, 
                                   CityscapesSemSegEvaluator, 
                                   CityscapesInstanceEvaluator)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import PanopticDeeplabDatasetMapper, add_panoptic_deeplab_config
from Validation_Loss_Hook import LossEvalHook
from detectron2.engine import hooks
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from Custom_Tensor_board import CustomTensorboardXWriter

def build_sem_seg_train_aug(cfg):
    """
        Augumentation function. Add all the required augumentations to be prfomed 
        on train images to aug list
    """
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]

    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    # augs.append(T.RandomRotation(angle= [30.0, 360.0], sample_style= 'range' ))
    return augs

class CustomTrainer(DefaultTrainer):
    """
    Derived from "DefaultTrainer" which contains pre-defined logic for standard
    Training work flow. Override the methods of base class to implement own training logic
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations= build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     mapper = PanopticDeeplabDatasetMapper(cfg, augmentations = [])
    #     return build_detection_test_loader(cfg, dataset_name, mapper= mapper)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Building learning rate scheduler
        """
        return build_lr_scheduler(cfg, optimizer)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        
    def build_hooks(self):
        hooks = super().build_hooks()
        print("-"*50)
        print("building_Validation hook for validation loss")
        hooks.insert(-1,LossEvalHook(
            20, #cfg.TEST.EVAL_PERIOD,   # Frequency of calcualtion - after every n iterations
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                PanopticDeeplabDatasetMapper(self.cfg, augmentations= build_sem_seg_train_aug(cfg))
            )
        ))

        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        # hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        Add the required evaluators in this method
        """

        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # if evaluator_type == "cityscapes_panoptic_seg":
        #     evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        #     evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # if evaluator_type == "coco_panoptic_seg":
        # # `thing_classes` in COCO panoptic metadata includes both thing and
        # # stuff classes for visualization. COCOEvaluator requires metadata
        # # which only contains thing classes, thus we map the name of
        # # panoptic datasets to their corresponding instance datasets.

        #      evaluator_list.append(
        #          COCOEvaluator(dataset_name, output_dir=output_folder)
        #         )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    
    def build_writers(self):
        """
        Overwrites the default writers to contain our custom tensorboard writer

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            CustomTensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
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

    output_image_folder_path = "/home/yalamaku/Documents/Thesis/result_images_for_report"

    if Visualization:
        #img_no = 1
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

                # cv2.imwrite(os.path.join(output_image_folder_path, str(img_no)+".png"), out.get_image()[:,:,::-1])
                # print(f"vis image saved to {output_image_folder_path}")
                # img_no = 1 + img_no

            # Close all windows
            cv2.destroyAllWindows()

    global cfg

    cfg = get_cfg()

    config_file = "/home/yalamaku/Documents/Thesis/my_scripts/detectron2/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml"
    
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.DATASETS.TRAIN = (data_name + "train",)
    cfg.DATASETS.TEST = (data_name + "valid",)
    cfg.TEST.EVAL_PERIOD = 100
    # cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "/home/yalamaku/Documents/Thesis/Detectron2_Downloaded_Weights/cityscapes_dsconv.pkl"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = total_classes-1
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.40
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.40
    cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = 512 #256 #512 #128
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # cfg.SOLVER.WEIGHT_DECAY =0.001
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.001
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.001
    # cfg.INPUT.MIN_SIZE_TRAIN = (256, 320, 384, 448, 512)
    # cfg.INPUT.MIN_SIZE_TRAIIN_SAMPLING = "choice"
    # cfg.INPUT.MIN_SIZE_TEST = 376
    # cfg.INPUT.MAX_SIZE_TRAIN = 2016
    # cfg.INPUT.MAX_SIZE_TEST = 672
    # cfg.INPUT.CROP.ENABLED = False
    # cfg.INPUT.CROP.SIZE = #(256, 512)
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.OUTPUT_DIR = '/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/Model_weights_2/train_10'


    # Uncomment this line If arguments are passed using command line
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        return res

    trainer = CustomTrainer(cfg)

    # # creates a hook that after each iter calculates the validation loss on the next batch
    # # Register the hoooks
    # trainer.register_hooks(
    #     [ValLossHook(cfg, cfg.DATASETS.TEST)]
    # )
    # # The PeriodicWriter needs to be the last hook, otherwise it wont have access to valloss metrics
    # # Ensure PeriodicWriter is the last called hook
    # periodic_writer_hook = [hook for hook in trainer._hooks if isinstance(hook, hooks.PeriodicWriter)]
    # all_other_hooks = [hook for hook in trainer._hooks if not isinstance(hook, hooks.PeriodicWriter)]
    # trainer._hooks = all_other_hooks + periodic_writer_hook

    # uncomment this line if resume config is provided from command line
    # trainer.resume_or_load(resume=args.resume)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":

    print(torch.__version__, torch.cuda.is_available())
    print("Is CUDA available:", torch.cuda.is_available())
    no_of_gpus = torch.cuda.device_count()
    print("Avialble no of GPU's :", no_of_gpus)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        1, #args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )