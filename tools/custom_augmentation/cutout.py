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
    
    def apply_box(self, boxes):
        #print("box:", boxes)
        result = []
        for box in boxes:
            bx0, by0, bx1, by1 = box
            r = box
            for x0, y0, x1, y1 in self.target_rects:
                if x0 <= bx0 and bx1 <= x1:
                    if y0 <= by0 and by1 <= y1:
                        r = [bx0, by0, bx0, by0]
                        break
                    elif y0 <= by0 and by0 <= y1:
                        r = [bx0, y1, bx1, by1]
                        break
                    elif y0 <= by1 and by1 <= y1:
                        r = [bx0, by0, bx1, y0]
                        break
                elif y0 <= by0 and by1 <= y1:
                    if x0 <= bx0 and bx0 <= x1:
                        r = [x1, by0, bx1, by1]
                        break
                    elif x0 <= bx1 and bx1 <= x1:
                        r = [bx0, by0, x0, by1]
                        break
            # 90% removal -> remove box
            if (bx1-bx0)*(by1-by0)*0.1 > (r[2]-r[0])*(r[3]-r[1]):
                r = [bx0, by0, bx0, by0]
            result.append(r)
        #print("after:", np.array(result))
        return np.array(result)

class Cutout(Augmentation):
    """
    Cutout.
    """

    def __init__(self, prob=0.5, *, size_pct=(0.02, 0.05), aspect=(0.33, 3), num=(5, 10)):
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
    input_args = ("image", "boxes")
    def __init__(self, prob=0.5, *, size_pct=(0.02, 0.05), aspect=(0.33, 3), num=(5, 10)):
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
    
    def _get_transform_rects(self, h, w, xoff, yoff, imgh, imgw):
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

                cx = np.random.randint(0, w) + xoff
                cy = np.random.randint(0, h) + yoff
                x0 = max(0,    cx - _w//2)
                x1 = min(imgw, cx + _w//2)
                y0 = max(0,    cy - _h//2)
                y1 = min(imgh, cy + _h//2)
                rects.append((x0, y0, x1, y1))
            return rects
        else:
            return []

    def get_transform(self, img, boxes):
        """boxes: XYXY_ABS instance boxes"""
        h, w = img.shape[:2]
        rects = []
        for (x1, y1, x2, y2) in boxes:
            rects.extend(self._get_transform_rects(y2-y1, x2-x1, x1, y1, h, w))
        if len(rects) == 0:
            return NoOpTransform()
        else:
            return CutoutTransform(rects)
