# coco.py

from typing import Union, Tuple, List, Dict
from pathlib import Path
import random
import itertools

from shapely.geometry import Polygon, MultiPolygon
import utils.other
import numpy as np


def train_test_split_coco(chips_stats: Dict) -> Tuple[List, List]:
    chips_list = list(chips_stats.keys())
    random.seed(1)
    random.shuffle(chips_list)
    split_idx = round(len(chips_list) * 0.2)  # 80% train, 20% test.
    train_split = chips_list[split_idx:]
    val_split = chips_list[:split_idx]

    # Apply split to geometries/stats.
    train_chip_dfs = {k: chips_stats[k] for k in sorted(train_split)}
    val_chip_dfs = {k.replace('train', 'val'): chips_stats[k] for k in sorted(val_split)}

    return train_chip_dfs, val_chip_dfs


def format_coco(set_: Dict, chip_width: int, chip_height: int):
    """
    Format extracted chip geometries to COCO json format.

    Coco train/val have specific ids, formatting requires the split data..
    """
    cocojson = {
        "info": {},
        "licenses": [],
        'categories': [{'supercategory': 'AgriculturalFields',
                        'id': 1,   # id needs to match category_id.
                        'name': 'agfields_singleclass'}]}

    for key_idx, key in enumerate(set_.keys()):
        if 'train' in key:
            chip_id = int(key[21:])
        elif 'val' in key:
            chip_id = int(key[19:])

        key_image = ({"file_name": f'{key}.jpg',
                      "id": int(chip_id),
                      "height": chip_width,
                      "width": chip_height})
        cocojson.setdefault('images', []).append(key_image)

        for row_idx, row in set_[key]['chip_df'].iterrows():
            # Convert geometry to COCO segmentation format:
            # From shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1, ..]].
            # The annotations were encoded by RLE, except for crowd region (iscrowd=1)
            coco_xy = list(itertools.chain.from_iterable((x, y) for x, y in zip(*row.geometry.exterior.coords.xy)))
            coco_xy = [round(xy, 2) for xy in coco_xy]
            # Add COCO bbox in format [minx, miny, width, height]
            bounds = row.geometry.bounds  # COCO bbox
            coco_bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]
            coco_bbox = [round(xy, 2) for xy in coco_bbox]

            key_annotation = {"id": key_idx,
                              "image_id": int(chip_id),
                              "category_id": 1,  # with multiple classes use "category_id" : row.reclass_id
                              "mycategory_name": 'agfields_singleclass',
                              "old_multiclass_category_name": row['rcl_lc_name'],
                              "old_multiclass_category_id": row['rcl_lc_id'],
                              "bbox": coco_bbox,
                              "area": row.geometry.area,
                              "iscrowd": 0,
                              "segmentation": [coco_xy]}
            cocojson.setdefault('annotations', []).append(key_annotation)

    return cocojson


def move_coco_val_images(val_chips_list, path_train_folder):
    """Move val chip images to val folder, applies train/val split on images"""
    out_folder = path_train_folder.parent / 'val2016'
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for chip in val_chips_list:
        Path(rf'{path_train_folder}\{chip.replace("val", "train")}.jpg').replace(rf'{out_folder}\{chip}.jpg')


def coco_to_shapely(fp_coco_json: Union[Path, str],
                    categories: List[int]=None) -> Dict:
    """
    Transforms coco json annotations to shapely format.

    Args:
        fp_coco_json: Input filepath coco json file.
        categories: Categories will filter to specific categories and images that contain at least one
        annotation of that category.

    Returns:
        Dictionary of image key and shapely Multipolygon
    """

    data = utils.other.load_saved(fp_coco_json, file_format='json')
    if categories is not None:
        # Get image ids/file names that contain at least one annotation of the selected categories.
        image_ids = list(set([x['image_id'] for x in data['annotations'] if x['category_id'] in categories]))
    else:
        image_ids = list(set([x['image_id'] for x in data['annotations']]))
    file_names = [x['file_name'] for x in data['images'] if x['id'] in image_ids]

    # Extract selected annotations per image.
    extracted_geometries = {}
    for image_id, file_name in zip(image_ids, file_names):
        annotations = [x for x in data['annotations'] if x['image_id'] == image_id]
        # Filter to annotations of the selected category.
        annotations = [x for x in annotations if x['category_id'] in categories]
        segments = [segment['segmentation'][0] for segment in annotations]  # format [x,y,x1,y1,...]

        # Create shapely Multipolygons from COCO format polygons.
        mp = MultiPolygon([Polygon(np.array(segment).reshape((int(len(segment) / 2), 2))) for segment in segments])
        extracted_geometries[str(file_name)] = mp

    return extracted_geometries


