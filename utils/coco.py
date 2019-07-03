# coco.py

from typing import Union, Tuple, List, Dict
from pathlib import Path
import random
import itertools

from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from PIL import Image as pilimage

import utils.other


def train_test_split(chip_dfs: Dict, test_size=0.2, seed=1) -> Tuple[Dict, Dict]:
    """Split chips into training and test set.

    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        test_size: Relative number of chips to be put in the test dataset. 1-test_size is the size of the
        training data set.
    """
    chips_list = list(chip_dfs.keys())
    random.seed(seed)
    random.shuffle(chips_list)
    split_idx = round(len(chips_list) * test_size)
    train_split = chips_list[split_idx:]
    val_split = chips_list[:split_idx]

    train_chip_dfs = {k: chip_dfs[k] for k in sorted(train_split)}
    val_chip_dfs = {k.replace('train', 'val'): chip_dfs[k] for k in sorted(val_split)}

    return train_chip_dfs, val_chip_dfs


def format_coco(chip_dfs: Dict, chip_width: int, chip_height: int):
    """Format train and test chip geometries to COCO json format.

    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        chip_width: width of the chip in pixel size.
        chip_height: height of the chip in pixel size.

    COCOjson example structure and instructions below. For more detailed information on building a COCO
        dataset see http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    cocojson = {
        "info": {...},
        "licenses": [...],
        "categories": [{"supercategory": "person","id": 1,"name": "person"},
                       {"supercategory": "vehicle","id": 2,"name": "bicycle"},
                       ...],
        "images":  [{"file_name": "000000289343.jpg", "height": 427, "width": 640, "id": 397133},
                    {"file_name": "000000037777.jpg", "height": 230, "width": 352, "id": 37777},
                    ...],
        "annotations": [{"segmentation": [[510.66,423.01,...,510.45,423.01]], "area": 702.10, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                        {"segmentation": [[340.32,758.01,...,134.25,875.01]], "area": 342.08, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                         ...]
        }

    - "id" in "categories" has to match "category_id" in "annotations".
    - "id" in "images" has to match "image_id" in "annotations".
    - "segmentation" in "annotations" is encoded in Run-Length-Encoding (except for crowd region (iscrowd=1)).
    - "id" in "annotations has to be unique for each geometry, so 4370 geometries in 1000 chips > 4370 unique
       geometry ids. However, does not have to be unique between coco train and validation set.
    - "file_name" in "images" does officially not have to match the "image_id" in "annotations" but is strongly
       recommended.
    """
    cocojson = {
        "info": {},
        "licenses": [],
        'categories': [{'supercategory': 'AgriculturalFields',
                        'id': 1,  # needs to match category_id.
                        'name': 'agfields_singleclass'}]}

    annotation_id = 1

    for chip_name in chip_dfs.keys():

        if 'train' in chip_name:
            chip_id = int(chip_name[21:])
        elif 'val' in chip_name:
            chip_id = int(chip_name[19:])

        image = {"file_name": f'{chip_name}.jpg',
                  "id": int(chip_id),
                  "height": chip_width,
                  "width": chip_height}
        cocojson.setdefault('images', []).append(image)

        for _, row in chip_dfs[chip_name]['chip_df'].iterrows():
            # Convert geometry to COCO segmentation format:
            # From shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1, ..]].
            # The annotations were encoded by RLE, except for crowd region (iscrowd=1)
            coco_xy = list(itertools.chain.from_iterable((x, y) for x, y in zip(*row.geometry.exterior.coords.xy)))
            coco_xy = [round(coords, 2) for coords in coco_xy]
            # Add COCO bbox in format [minx, miny, width, height]
            bounds = row.geometry.bounds  # COCO bbox
            coco_bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]
            coco_bbox = [round(coords, 2) for coords in coco_bbox]

            annotation = {"id": annotation_id,
                           "image_id": int(chip_id),
                           "category_id": 1,  # with multiple classes use "category_id" : row.reclass_id
                           "mycategory_name": 'agfields_singleclass',
                           "old_multiclass_category_name": row['r_lc_name'],
                           "old_multiclass_category_id": row['r_lc_id'],
                           "bbox": coco_bbox,
                           "area": row.geometry.area,
                           "iscrowd": 0,
                           "segmentation": [coco_xy]}
            cocojson.setdefault('annotations', []).append(annotation)

            annotation_id += 1

    return cocojson


def move_coco_val_images(inpath_train_folder, val_chips_list):
    """Move validation chip images to val folder (applies train/val split on images)

    Args:
        inpath_train_folder: Filepath to the training COCO image chip "train" folder
        val_chips_list: List of validation image key names that should be moved.
    """
    outpath_val_folder = inpath_train_folder.parent / 'val2016'
    Path(outpath_val_folder).mkdir(parents=True, exist_ok=True)
    for chip in val_chips_list:
        Path(rf'{inpath_train_folder}\{chip.replace("val", "train")}.jpg').replace(rf'{outpath_val_folder}\{chip}.jpg')


def coco_to_shapely(inpath_json: Union[Path, str],
                    categories: List[int] = None) -> Dict:
    """Transforms COCO annotations to shapely geometry format.

    Args:
        inpath_json: Input filepath coco json file.
        categories: Categories will filter to specific categories and images that contain at least one
        annotation of that category.

    Returns:
        Dictionary of image key and shapely Multipolygon.
    """

    data = utils.other.load_json(inpath_json)
    if categories is not None:
        # Get image ids/file names that contain at least one annotation of the selected categories.
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations'] if x['category_id'] in categories])))
    else:
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations']])))
    file_names = [x['file_name'] for x in data['images'] if x['id'] in image_ids]

    # Extract selected annotations per image.
    extracted_geometries = {}
    for image_id, file_name in zip(image_ids, file_names):
        annotations = [x for x in data['annotations'] if x['image_id'] == image_id]
        if categories is not None:
            annotations = [x for x in annotations if x['category_id'] in categories]

        segments = [segment['segmentation'][0] for segment in annotations]  # format [x,y,x1,y1,...]

        # Create shapely Multipolygons from COCO format polygons.
        mp = MultiPolygon([Polygon(np.array(segment).reshape((int(len(segment) / 2), 2))) for segment in segments])
        extracted_geometries[str(file_name)] = mp

    return extracted_geometries


def plot_coco(inpath_json, inpath_image_folder, start=0, end=2):
    """Plot COCO annotations and image chips"""
    extracted = utils.coco.coco_to_shapely(inpath_json)

    for key in sorted(extracted.keys())[start:end]:
        print(key)
        plt.figure(figsize=(5, 5))
        plt.axis('off')

        img = np.asarray(pilimage.open(rf'{inpath_image_folder}\{key}'))
        plt.imshow(img, interpolation='none')

        mp = extracted[key]
        patches = [PolygonPatch(p, ec='r', fill=False, alpha=1, lw=0.7, zorder=1) for p in mp]
        plt.gca().add_collection(PatchCollection(patches, match_original=True))
        plt.show()
