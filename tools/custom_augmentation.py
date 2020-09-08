import numpy as np
import sys
from fvcore.transforms.transform import (
    NoOpTransform,
    Transform,
)

from PIL import Image, ImageDraw
from detectron2.augmentation import Augmentation

class CutoutTransform(Transform):
    def __init__(self, target_rects):
        """
        Args:
            target_rects (list[(x0, y0, x1, y1)]): target coordinates
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        draw = ImageDraw.Draw(img)
        for rect in self.target_rects:
            draw.rectangle(rect, fill=(0,0,0))
        return img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

class Cutout(Augmentation):
    """
    Cutout.
    """

    def __init__(self, prob=0.5, *, size_pct=0.1, num=10):
        """
        Args:
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
            size_pct (float): the length ratio of shorter box of cutouts.
            num (int): the maximum number of boxes
        """
        super().__init__()
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        assert 0.0 <= size_pct <= 1.0, f"Pct must be between 0.0 and 1.0 (given: {size_pct})"
        self.prob = prob
        self.size_pct = size_pct
        self.num = num
    
    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            num = np.random.randint(0, self.num+1)
            length = int(size_pct * min(h, w))
            rects = []
            for _ in range(num):
                x0 = np.random.randint(0, w)
                x1 = min(x0 + length, w)
                y0 = np.random.randint(0, h)
                y1 = min(y0 + length, h)
                rects.append((x0, y0, x1, y1))
            return CutoutTransform(rects)
        else:
            return NoOpTransform()
    
            

