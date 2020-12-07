import numpy as np
import sys
from fvcore.transforms.transform import (
    NoOpTransform,
    Transform,
)

from PIL import Image
from detectron2.data.transforms.augmentation import Augmentation

import ace.config
from ace import ace_helpers
from .cutout import CutoutTransform

class CutoutCAV(Augmentation):
    """
    Cutout according to CAV
    """
    def __init__(self, config_path: str, prob=0.5, *, config_overrides: [str]=[], size_pct=(0.02, 0.05), aspect=(0.33, 3)):
        """
        Args:
            config_path (str): configuration file of ACE model
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
            config_overrides (dict): override options for the config
            size_pct (float or tuple[float]): the area ratio of boxes w.r.t image..
            aspect (float or tuple[float]): aspect ratio of boxes. height/width
        """
        super().__init__()
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        if type(size_pct) is tuple:
            assert 0.0 <= size_pct[0] <= 1.0, f"Pct must be between 0.0 and 1.0 (given: {size_pct})"
            assert 0.0 <= size_pct[1] <= 1.0, f"Pct must be between 0.0 and 1.0 (given: {size_pct})"
        else:
            assert 0.0 <= size_pct <= 1.0, f"Pct must be between 0.0 and 1.0 (given: {size_pct})"

        self.prob = prob
        self.size_pct = size_pct if type(size_pct) is tuple else (size_pct, size_pct)
        self.size_pct_scale = self.size_pct[1] - self.size_pct[0]
        self.aspect = aspect if type(aspect) is tuple else (aspect, aspect)
        self.aspect_scale = self.aspect[1] - self.aspect[0]
        self.ace_config = ace.config.load(config_path, config_overrides)
        print("loading config:", self.ace_config)
        self.model = ace_helpers.make_model(self.ace_config.model)
        self.bn = 'res5_0'
        print("bottleneck:", self.bn)

    def get_transform(self, img):
        img_acts = self.model.run_imgs([img[:,:,::-1]], self.bn)
        print(img.shape, img_acts[0].shape)
        return NoOpTransform()
