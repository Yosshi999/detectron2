from tqdm import tqdm
import json
from pathlib import Path

CITYPERSONS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pedestrian"},
    {"color": [  0,  0,142], "isthing": 1, "id": 2, "name": "rider"},
    {"color": [107,142, 35], "isthing": 1, "id": 3, "name": "sitting person"},
    {"color": [190,153,153], "isthing": 1, "id": 4, "name": "person (other)"},
]

def citypersons2coco_detection(image_root: str, gt_root: str, out_fn: str, use_vis=True):
    valid_cats = [x['name'] for x in CITYPERSONS_CATEGORIES]
    cat2id = { x['name']: x['id'] for x in CITYPERSONS_CATEGORIES }

    IMAGE = Path(image_root)
    GT = Path(gt_root)
    city_paths = sorted(GT.iterdir())

    files = []
    for cp in city_paths:
        cityname = cp.stem
        for fp in sorted(cp.iterdir()):
            files.append( (cityname, fp) )
    print("{} annotation files".format(len(files)))

    images = []
    annotations = []
    idcounter = 0
    counter = 0
    for cityname, fp in tqdm(files):
        counter += 1
        with fp.open() as f:
            data = json.load(f)
        assert 'gtBboxCityPersons.json' in fp.name
        image = {
            'file_name': cityname + '/' + fp.name.replace('gtBboxCityPersons.json', 'leftImg8bit.png'),
            'height': data['imgHeight'],
            'width': data['imgWidth'],
            'id': counter
        }

        empty_image = True
        for i, lab in enumerate(data['objects']):
            cat = lab['label']
            if cat in valid_cats:
                empty_image = False
                x,y,w,h = lab['bboxVis'] if use_vis else lab['bbox']
                annotation = {
                    'iscrowd': 0,
                    'image_id': counter,
                    'bbox': [x,y,w,h],
                    'area': float(w * h),
                    'category_id': cat2id[cat],
                    'ignore': 0,
                    'id': idcounter,
                    'segmentation': [[x, y, x, y+h, x+w, y+h, x+w, y]]
                }
                annotations.append(annotation) 
                idcounter += 1
        if not empty_image:
            images.append(image)
    print('saving...')
    json_string = json.dumps({
        'categories': [ {'supercategory': 'none', 'id': x['id'], 'name': x['name']} for x in CITYPERSONS_CATEGORIES ],
        'images': images,
        'annotations': annotations,
        'type': 'instances'
    })
    with open(out_fn, 'w') as f:
        f.write(json_string)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('gt', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()
    citypersons2coco_detection(args.image, args.gt, args.out)
