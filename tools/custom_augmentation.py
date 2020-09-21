import numpy as np
import sys
from fvcore.transforms.transform import (
    NoOpTransform,
    Transform,
)

from PIL import Image
from detectron2.data.transforms.augmentation import Augmentation

class CutoutTransform(Transform):
    def __init__(self, target_rects):
        """
        Args:
            target_rects (list[(x0, y0, x1, y1)]): target coordinates
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, _img):
        img = _img.copy()
        for rect in self.target_rects:
            x0, y0, x1, y1 = rect
            img[y0:y1, x0:x1] = 0
        return img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

class Cutout(Augmentation):
    """
    Cutout.
    """

    def __init__(self, prob=0.5, *, size_pct=(0.02, 0.2), aspect=(0.33, 3), num=1):
        """
        Args:
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
            size_pct (float or tuple[float]): the area ratio of boxes w.r.t image..
            aspect (float or tuple[float]): aspect ratio of boxes. height/width
            num (int or tuple[int]): the maximum number of boxes
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
        self.num = num if type(num) is tuple else (num, num)
    
    def get_transform(self, img):
        h, w = img.shape[:2]
        area = h * w
        do = self._rand_range() < self.prob
        if do:
            num = np.random.randint(self.num[0], self.num[1]+1)
            rects = []
            for _ in range(num):
                _size_pct = np.random.rand() * self.size_pct_scale + self.size_pct[0]
                _area = area * _size_pct
                _aspect = np.random.rand() * self.aspect_scale + self.aspect[0]
                _h = int(np.sqrt(_area * _aspect))
                _w = int(np.sqrt(_area / _aspect))

                x0 = np.random.randint(0, w)
                x1 = min(x0 + _w, w)
                y0 = np.random.randint(0, h)
                y1 = min(y0 + _h, h)
                rects.append((x0, y0, x1, y1))
            return CutoutTransform(rects)
        else:
            return NoOpTransform()
    
class ObjectAwareCutout(Augmentation):
    """
    Region aware cutout.
    """
    def __init__(self, prob=0.5, *, size_pct=0.1, num=5):
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
        raise NotImplementedError

