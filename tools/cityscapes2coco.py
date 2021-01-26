import json
from tqdm import tqdm
from typing import List, Tuple
from detectron2.data.datasets import load_cityscapes_instances
from detectron2.structures import BoxMode
from cityscapesscripts.helpers.labels import labels

labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
CITYSCAPES_CATEGORIES = [{'id': idx, 'name': label.name} for idx, label in enumerate(labels)]

"""load_cityscapes_instances(image_dir: str, gt_dir: str) ->
[
    {
        file_name: str,
        image_id: str, # os.path.basename(file_name)
        height: int,
        width: int,
        annotations: [{
            iscrowd: bool,
            label: str, # category or f'{category}group'
            category_id: int,
            segmentation: list[int],
            bbox: (int, int, int, int),
            bbox_mode: BoxMode # BoxMode.XYXY_ABS from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py#L224
        }]
    }
]
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help="path to image dir of cityscapes")
    parser.add_argument('gt', type=str, help="path to gt dir of cityscapes")
    parser.add_argument('out', type=str, help="output json")
    args = parser.parse_args()
    image_dir = args.image
    gt_dir = args.gt
    out_fn = args.out
    data = load_cityscapes_instances(image_dir, gt_dir)

    images = []
    annotations = []
    idcounter = 0
    counter = 0
    for dict_per_image in tqdm(data):
        counter += 1
        images.append({
            'file_name': dict_per_image['file_name'],
            'height': dict_per_image['height'],
            'width': dict_per_image['width'],
            'id': counter
        })
        for anno in dict_per_image["annotations"]:
            idcounter += 1
            assert anno['bbox_mode'] == BoxMode.XYXY_ABS
            x1,y1,x2,y2 = anno['bbox']
            w = x2 - x1
            h = y2 - y1
            annotations.append({
                'iscrowd': int(anno['iscrowd']),
                'image_id': counter,
                'bbox': [x1, y1, w, h],
                'area': float(w * h),
                'category_id': anno['category_id'],
                'segmentation': anno['segmentation'],
                'ignore': 0,
                'id': idcounter
            })

    with open(out_fn, 'w') as f:
        f.write(json.dumps({
            'categories': [{
                'supercategory': 'none',
                'id': x['id'],
                'name': x['name']
            } for x in CITYSCAPES_CATEGORIES],
            'images': images,
            'annotations': annotations,
            'type': 'instances'
        }))
