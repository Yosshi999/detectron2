import os

from detectron2.data.datasets import register_coco_instances
#from citypersons2coco import CITYPERSONS_CATEGORIES

data_dir = '/home/u00215/data/BDD'
for split in 'train', 'val':
    json_file = os.path.join(data_dir, 'bdd100k_labels_images_det_coco_' + split + '.json')
    image_root = os.path.join(data_dir, 'bdd100k', 'images', '100k', split)
    register_coco_instances('bdd_' + split, {}, json_file, image_root)

#data_dir = '/home/u00172/git/CenterNet/data/kitti'
#for SPLIT in '3dop', 'subcnn':
#    for split in 'train', 'val':
#        json_file = os.path.join(data_dir, 'annotations_detectron2', 'kitti_{}_{}.json'.format(SPLIT, split))
#        image_root = os.path.join(data_dir, 'training', 'image_2')
#        register_coco_instances('kitti_{}_{}'.format(SPLIT, split), {}, json_file, image_root)

for split in 'train', 'val':
    json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cocoformats/citypersons_det_coco_' + split + '.json')
    image_root = os.path.join('/home/u00215/data/cityscapes/leftImg8bit', split)
    register_coco_instances('citypersons_' + split, {}, json_file, image_root)
