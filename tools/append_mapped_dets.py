import custom

from pycocotools.coco import COCO
from detectron2.data import MetadataCatalog
from tqdm import tqdm

import argparse
from pathlib import Path
import json

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="dataset id registered to detectron2 (must be cocoformat)")
	parser.add_argument("target", help="path to result det json")
	args = parser.parse_args()

	target = Path(args.target)
	SUFFIX = "_moreInfo"
	outfile = target.parent / (target.stem + SUFFIX + ".json")

	meta = MetadataCatalog.get(args.dataset)
	assert hasattr(meta, "json_file")
	api = COCO(meta.json_file)

	outdets = []
	with target.open("r") as f:
		dets = json.load(f)
		for det in tqdm(dets):
			det['name'] = api.imgs[det['image_id']]
			det['category'] = api.cats[det['category_id']]
			outdets.append(det)
	with outfile.open("w") as f:
		f.write(json.dumps(outdets))
		f.flush()
