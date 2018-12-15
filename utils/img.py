# img.py

from typing import Tuple, Generator, Union

import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import Polygon
import affine
import warnings
from pathlib import Path

import utils.geo
import itertools
import numpy as np
import rasterio
import shapely
from shapely.geometry import Polygon
from PIL import Image as pilimg
from skimage import exposure, img_as_ubyte
from tqdm import tqdm


def get_chip_windows(raster_width: int,
                     raster_height: int,
                     raster_transform,
                     chip_width: int=256,
                     chip_height: int=256,
                     skip_partial_chips: bool=False,
                     ) -> Generator[Tuple[Window, Polygon, affine.Affine], any, None]:
    """Generator for rasterio windows of specified pixel size to iterate over an image in chips.

    Chips are created row wise, from top to bottom of the raster.

    Args:
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        skip_partial_chips: Skip image chips at the edge of the raster that do not result in a full size chip.

    Returns : Yields tuple of rasterio window, Polygon and transform.

    """

    big_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    col_row_offsets = itertools.product(range(0, raster_width, chip_width), range(0, raster_height, chip_height))

    for col_off, row_off in col_row_offsets:

        chip_window = Window(col_off=col_off, row_off=row_off, width=chip_width, height=chip_height)

        if skip_partial_chips:
            if row_off + chip_height > raster_height or col_off + chip_width > raster_width:
                continue

        chip_window = chip_window.intersection(big_window)
        chip_transform = rasterio.windows.transform(chip_window, raster_transform)
        chip_bounds = rasterio.windows.bounds(chip_window, raster_transform)  # Uses transform of full raster.
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_transform, chip_poly)


def cut_chip_geometries(vector_df, raster_width, raster_height, raster_transform, chip_width=128, chip_height=128,):
    """Workflow to cut a vector geodataframe to chip geometries.

    Filters small polygons and skips empty chips.

    Args:
        vector_df: Geodataframe containing the geometries to be cut to chip geometries.
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.

    Returns: Dictionary containing the final chip_df, chip_window, chip_transform, chip_poly objects.
    """

    generator_window_bounds = get_chip_windows(raster_width=raster_width,
                                               raster_height=raster_height,
                                               raster_transform=raster_transform,
                                               chip_width=chip_width,
                                               chip_height=chip_height,
                                               skip_partial_chips=True)

    all_chip_dfs = {}
    for i, (chip_window, chip_transform, chip_poly) in enumerate(tqdm(generator_window_bounds)):

        # # Clip geometry to chip
        chip_df = vector_df.pipe(utils.geo.clip, clip_poly=chip_poly, keep_biggest_poly_=True)
        if not all(chip_df.geometry.is_empty):
            chip_df.geometry = chip_df.simplify(1, preserve_topology=True)
        else:
            continue
        # Drop small geometries
        chip_df = chip_df[chip_df.geometry.area * (10 * 10) > 5000]  #5000 sqm in UTM
        # Transform to chip pixelcoordinates and invert y-axis for COCO format.
        if not all(chip_df.geometry.is_empty):
            chip_df = chip_df.pipe(utils.geo.to_pixelcoords, reference_bounds=chip_poly.bounds, scale=True,
                                   ncols=chip_width, nrows=chip_height)
            chip_df = chip_df.pipe(utils.geo.invert_y_axis, reference_height=chip_height)
        else:
            continue

        chip_name = f'COCO_train2016_000000{100000+i}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'
        all_chip_dfs[chip_name] = {'chip_df': chip_df,
                                   'chip_window': chip_window,
                                   'chip_transform': chip_transform,
                                   'chip_poly': chip_poly}
    return all_chip_dfs


def cut_chip_images(img_path, chip_windows, out_folder, bands=[3, 2, 1]):
    """Cuts image chips and exports them

    Args:
        img_path:
        chip_windows:
        out_folder:
        bands:

    Returns:

    """
    src = rasterio.open(img_path)

    all_chip_stats = {}
    for i, chip_window in enumerate(chip_windows):
        img_array = np.dstack(list(src.read(bands, window=chip_window)))
        img_array = exposure.rescale_intensity(img_array, in_range=(0, 2200))  # Sentinel2 range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # Export chip images
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        chip_name = f'COCO_train2016_000000{100000+i}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'
        with open(Path(rf'{out_folder}\{chip_name}.jpg'), 'w') as dst:
            img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        all_chip_stats[chip_name] = {'mean': img_array.mean(axis=(0, 1)),
                                     'std': img_array.std(axis=(0, 1))}

    src.close()

    return all_chip_stats
