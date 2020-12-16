"""custom dataset mapper for box transform"""
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

class CustomDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # you don't need segmentation

        assert "annotations" in dataset_dict
        boxes = []
        for obj in dataset_dict["annotations"]:
            box = BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            boxes.append(box)
        aug_input = T.StandardAugInput(image, boxes=boxes)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, boxes = aug_input.image, aug_input.boxes

        image_shape = image.shape[:2] # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # you don't need pre-computed proposals

        if not self.is_train:
            dataset_dict.pop("anotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        assert not self.use_instance_mask
        assert not self.use_keypoint
        for anno in dataset_dict["annotations"]:
            anno.pop("segmentation", None)
            anno.pop("keypoints", None)

        annos = []
        for obj, bbox in zip(dataset_dict.pop("annotations"), boxes):
            if not obj.get("iscrowd", 0) == 0: continue
            obj["bbox"] = np.minimum(bbox, list(image_shape + image_shape)[::-1])
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(obj)
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        assert not self.recompute_boxes
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
