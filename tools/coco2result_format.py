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
	parser.add_argument("--out", default="../cocoformats", help="path to output folder")
	args = parser.parse_args()

	SUFFIX = "_in_result_format"
	outfile = Path(args.out) / (args.dataset + SUFFIX + ".json")

	meta = MetadataCatalog.get(args.dataset)
	assert hasattr(meta, "json_file")
	api = COCO(meta.json_file)

	outdets = []
	
	for i, name in enumerate(tqdm(api.imgs)):
		anns = api.getAnnIds([i])
		dets = api.loadAnns(anns)
		for det in dets:
			det['name'] = name
			det['category'] = api.cats[det['category_id']]['name']
			outdets.append(det)

	with outfile.open("w") as f:
		f.write(json.dumps(outdets))
		f.flush()

