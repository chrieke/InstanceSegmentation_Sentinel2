# Pre-processing

# This workflow pre-processes Sentinel-2 data to image chips and LPIS field geometries to matching COCO json
# format for use with the FCIS model.
# Imagery and geometries for the full Denmark aoi of the thesis are several gigabytes in file size. I have included
# data for a small subset roi for demonstration purposes. The full 2016 LPIS field data set can be downloaded here:
# https://kortdata.fvm.dk/download/Markblokke_Marker?page=MarkerHistoriske
#
# ![Subset roi](msc_codeshare/test_aoi_subset.png)


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'line_metar')
from pathlib import Path
import itertools
import numpy as np
import random
from typing import List, Dict, Tuple

import rasterio
import rasterio.plot
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from PIL import Image as pilimg
from skimage import exposure, img_as_ubyte
from tqdm import tqdm
import matplotlib.pyplot as plt
import cgeo
import scripts.preprocessing_helper as preproc_helper
import pprint


fp_s2 = Path(r'data\original\RGB_small_cor.tif')
fp_fields = Path(r'data\original\marker_small.shp')

# #  Geometry Pre-processing Pipeline
# Includes read/clean of geodataframe, clip to aoi, validify, reproject and clean geometries, reclassify labels.

with rasterio.open(fp_s2) as src:
    meta = src.meta
    meta.update({'bounds': src.bounds})


def preprocess_vector(inpath, meta):
    df = (gpd.read_file(str(inpath), encoding='utf-8')  # utf-8 for danish special characters.
             .rename(columns={'Afgroede': 'lcsub', 'AfgKode': 'lcsub_id'})
             .drop(['GB', 'GEOMETRISK', 'MARKNUMMER'], axis=1)
             .pipe(cgeo.geo.buffer_zero)
             .pipe(cgeo.geo.close_holes)
             .pipe(cgeo.geo.set_crs, 3044)
             .to_crs(meta['crs'])
             .pipe(cgeo.geo.clip, clip_poly=shapely.geometry.box(*meta['bounds']), explode_mp_=True)
             .pipe(cgeo.geo.reclassify_col, rcl_scheme=preproc_helper.reclass_legend, col_classlabels='lcsub',
                   col_classids='lcsub_id', drop_other_classes=True)
             .assign(geometry=lambda _df: _df.geometry.simplify(5, preserve_topology=True))
             .pipe(cgeo.geo.buffer_zero)
             .assign(area_sqm=lambda _df: _df.geometry.area)
             .pipe(cgeo.geo.reduce_precision, precision=4)
             .reset_index(drop=True)
             .assign(fid=lambda _df: range(0, len(_df.index))))
    return df


df = cgeo.other.read_or_new_save(path=Path(r'data\output_preproc\preprocessed_marker_small.pkl'),
                                 default_data=preprocess_vector,
                                 callable_args={'inpath': fp_fields, 'meta': meta})

print('df.info()', df.info())
# print('overall number of fields\n', len(df))
# print('\nnumber of fields per class\n', df.groupby(['rcl_lcsub']).fid.aggregate(len).sort_values(ascending=False))
# print('\noverall mean area of fields\n', df.area_sqm.mean())
# print('\nmean area of fields per class\n', df.groupby(['rcl_lcsub']).area_sqm.mean().sort_values(ascending=False))


# # Cut chips for images and geometries

def cut_chips(img_path, df, chip_width, chip_height, bands):
    chips_stats = {}
    src = rasterio.open(img_path)
    generator_window_bounds = cgeo.img.get_chip_windows(meta_raster=src.meta,
                                                        chip_width=chip_width,
                                                        chip_height=chip_height,
                                                        skip_partial_chips=True)

    for i, (chip_window, chip_poly, chip_transform) in enumerate(tqdm(generator_window_bounds)):
        if i % 100 == 0: print(i)

        # # Clip geometry to chip
        chip_df = df.pipe(cgeo.geo.clip, clip_poly=chip_poly, keep_biggest_poly_=True)
        if not all(chip_df.geometry.is_empty):
            chip_df.geometry = chip_df.simplify(1, preserve_topology=True)
        else:
            continue
        chip_df = chip_df[chip_df.geometry.area * (10 * 10) > 5000]
        # Transform to chip pixelcoordinates and invert y-axis for COCO format.
        if not all(chip_df.geometry.is_empty):
            chip_df = chip_df.pipe(cgeo.geo.to_pixelcoords, reference_bounds=chip_poly.bounds, scale=True,
                                   ncols=chip_width, nrows=chip_height)
            chip_df = chip_df.pipe(cgeo.geo.invert_y_axis, reference_height=chip_height)
        else:
            continue

        # # Clip image to chip
        img_array = np.dstack(list(src.read(bands, window=chip_window)))
        img_array = exposure.rescale_intensity(img_array, in_range=(0, 2200))  # Sentinel2 range.
        img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # # Export image chip.
        for folder in ['train2014', 'val2014']:
            Path(rf'data\output_preproc\chips\{folder}').mkdir(parents=True, exist_ok=True)
        chip_file_name = f'COCO_train2014_000000{100000+i}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'
        with open(Path(rf'data\output_preproc\chips\train2014\{chip_file_name}.jpg'), 'w') as dst:
            img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        # # Gather image statistics
        chips_stats[chip_file_name] = {'chip_df': chip_df,
                                       'mean': img_array.mean(axis=(0, 1)),
                                       'std': img_array.std(axis=(0, 1))}
    src.close()
    return chips_stats


chip_width, chip_height = 128, 128
bands = [3, 2, 1]
chips_stats = cgeo.other.read_or_new_save(path=Path(r'data\output_preproc\chips_geo_stats.pkl'),
                                          default_data=cut_chips,
                                          callable_args={'img_path': fp_s2,
                                                         'df': df,
                                                         'chip_width': chip_width,
                                                         'chip_height': chip_height,
                                                         'bands': bands})

print('len(chips_stats)', len(chips_stats))


def train_test_split_coco(chips_stats: Dict) -> Tuple[List, List]:
    chips_list = list(chips_stats.keys())
    random.seed(1)
    random.shuffle(chips_list)
    split_idx = round(len(chips_list) * 0.2)  # 80% train, 20% test.
    chips_train = chips_list[split_idx:]
    chips_val = chips_list[:split_idx]

    return chips_train, chips_val


def apply_split_coco(chips_train: List, chips_val: List) -> Tuple[Dict, Dict]:
    # Apply split to geometries/stats.
    stats_train = {k: chips_stats[k] for k in sorted(chips_train)}
    stats_val = {k.replace('train', 'val'): chips_stats[k] for k in sorted(chips_val)}
    print(len(chips_train), len(chips_val))

    # Apply split to image chips: Move val chip images.
    for chip in chips_val:
        destination = Path(r"data\output_preproc\chips\val2014\{}.jpg".format(chip.replace('train', 'val')))
        Path(rf"data\output_preproc\chips\train2014\{chip}.jpg").replace(destination)

    return stats_train, stats_val


def format_cocojson(set_: Dict):
    """
    Format extracted chip geometries to COCO json format.

    Coco train/val have specific ids, formatting requires the split data..
    """
    cocojson = {
        "info": {},
        "licenses": [],
        'categories': [{'supercategory': 'AgriculturalFields', 'id': 1, 'name': 'agfields_singleclass'}]}
    # id needs to match category_id.

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
            # Convert poly to COCO segmentation format, from shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1,
            # ..]]. Except for crowd region (iscrowd=1), the annotations were encoded by RLE.
            coco_xy = list(itertools.chain.from_iterable((x, y) for x, y in zip(*row.geometry.exterior.coords.xy)))
            coco_xy = [round(xy, 2) for xy in coco_xy]
            # Add bbox.
            bounds = row.geometry.bounds  # COCO bbox format [minx, miny, width, height]
            coco_bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]
            coco_bbox = [round(xy, 2) for xy in coco_bbox]

            key_annotation = {"id": key_idx,
                              "image_id": int(chip_id),
                              "category_id": 1,  # with multiclass "category_id" : row.reclass_lcsub_id
                              "mycategory_name": 'agfields_singleclass',
                              "old_multiclass_category_name": row['rcl_lcsub'],
                              "old_multiclass_category_id": row['rcl_lcsub_id'],
                              "bbox": coco_bbox,
                              "area": row.geometry.area,
                              "iscrowd": 0,
                              "segmentation": [coco_xy]}
            cocojson.setdefault('annotations', []).append(key_annotation)

    return cocojson


outpath_cocojson_train = Path(r'data\output_preproc\chips\train2014.json')
outpath_cocojson_val = Path(r'data\output_preproc\chips\val2014.json')

if outpath_cocojson_train.exists() and outpath_cocojson_val.exists():
    cocojson_train = cgeo.other.read_saved(outpath_cocojson_train, file_format='json')
    cocojson_val = cgeo.other.read_saved(outpath_cocojson_val, file_format='json')
else:
    chips_train, chips_val = train_test_split_coco(chips_stats)
    stats_train, stats_val = apply_split_coco(chips_train, chips_val)
    cocojson_train = format_cocojson(stats_train)
    cocojson_val = format_cocojson(stats_val)
    cgeo.other.new_save(outpath_cocojson_train, cocojson_train, file_format='json')
    cgeo.other.new_save(outpath_cocojson_val, cocojson_val, file_format='json')


# # Gather chip statistics.
# statistics = {
#     'nr_chips': len(chips_stats.keys()),
#     'nr_chips_train': len(chips_train),
#     'nr_chips_val': len(chips_val),
#
#     'nr_polys': sum([len(df['chip_df']) for df in chips_stats.values()]),
#     'avg_polys_per_chip': sum([len(df['chip_df']) for df in chips_stats.values()]) / len(chips_stats.keys()),
#     'nr_polys_train': sum([len(df['chip_df']) for df in [chips_stats[key] for key in chips_train]]),
#     'nr_polys_val': sum([len(df['chip_df']) for df in [chips_stats[key] for key in chips_val]]),
#
#     'train_rgb_mean': list(np.asarray([df['mean'] for df in [chips_stats[key] for key in chips_train]]).mean(axis=0)),
#     'train_rgb_stdn': list(np.asarray([df['std'] for df in [chips_stats[key] for key in chips_train]]).mean(axis=0)),
#     # 'polys_classstats': afterchip_df.groupby(['reclass_lcsub']).object_id.aggregate(len).sort_values(ascending=False),
#     # 'polys_mean_sqm': afterchip_df.areasqm.mean(),
#     # 'polys_classareastats:': afterchip_df.groupby(['reclass_lcsub']).areasqm.mean(),
# }
#
# statistics = cgeo.other.read_or_new_save(path=Path(r'data\output_preproc\statistics.json'),
#                                          default_data=statistics,
#                                          file_format='json')
# pprint.pprint(statistics)
