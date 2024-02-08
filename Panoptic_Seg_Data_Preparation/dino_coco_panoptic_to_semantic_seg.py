#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image
import argparse

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument( "--panoptic_json", help= "path to coco panoptic annotations json file",
                        type= str, dest= 'panoptic_json')
    parser.add_argument( "--panoptic_root", help= "path to coco panoptic ground truth images",
                        type= str, dest= 'panoptic_root')
    parser.add_argument( "--sem_seg_root", help= "Path to output the semantic segs",
                        type= str, dest= 'sem_seg_root')
    parser.add_argument( "--categories_json", help= """category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1""", type= str, dest= 'categories_json')
    
    return parser.parse_args()
    




if __name__ == "__main__":
    # dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    # for s in ["val2017", "train2017"]:
    #     separate_coco_semantic_from_panoptic(
    #         os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
    #         os.path.join(dataset_dir, "panoptic_{}".format(s)),
    #         os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
    #         COCO_CATEGORIES,
    #     )

    # args = arg_parser()
    
    # separate_coco_semantic_from_panoptic(args[0], args[1], args[2], args[3])

    panoptic_json = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/train/train_coco_panoptic.json"
    panoptic_root = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/train/train_coco_panoptic"
    sem_seg_root = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/train/train_Dino_semantic_segmentation_masks"
    categories_json = "/home/yalamaku/Documents/Detectron_2/Dataset/ZED_camera_dataset/Roboflow_annotated_data/Categories.json"

    with open(categories_json, 'r') as f:
        categories_dict = json.load(f)

    print(type(categories_dict))
    print(len(categories_dict))
    print(categories_dict)

    separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories_dict)
