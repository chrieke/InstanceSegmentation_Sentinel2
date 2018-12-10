# coco.py

from typing import Union, List, Dict
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon
import utils.other
import numpy as np

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