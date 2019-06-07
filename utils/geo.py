# geo.py

import warnings
from typing import Union, Dict

import numpy as np
from geopandas import GeoDataFrame as GDF
from pandas import DataFrame as DF
import shapely
from shapely.geometry import Polygon
import rasterio.crs
import geopandas as gpd
from tqdm import tqdm

import utils.img


def buffer_zero(ingeo: Union[GDF, Polygon]) -> Union[GDF, Polygon]:
    """Make invalid polygons (due to self-intersection) valid by buffering with 0."""
    if isinstance(ingeo, Polygon):
        if ingeo.is_valid is False:
            return ingeo.buffer(0)
        else:
            return ingeo
    elif isinstance(ingeo, GDF):
        if False in ingeo.geometry.is_valid.unique():
            ingeo.geometry = ingeo.geometry.apply(lambda _p: _p.buffer(0))
            return ingeo
        else:
            return ingeo


def close_holes(ingeo: Union[GDF, Polygon]) -> Union[GDF, Polygon]:
    """Close polygon holes by limitation to the exterior ring."""
    def _close_holes(poly: Polygon):
        if poly.interiors:
            return Polygon(list(poly.exterior.coords))
        else:
            return poly

    if isinstance(ingeo, Polygon):
        return _close_holes(ingeo)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _close_holes(_p))
        return ingeo


def set_crs(df: GDF, epsg_code: Union[int, str]) -> GDF:
    """Sets dataframe crs in geopandas pipeline.

    TODO: Deprecate with next rasterio version that will integrate set_crs method.
    """
    df.crs = {'init': f'epsg:{str(epsg_code)}'}
    return df


def explode_mp(df: GDF) -> GDF:
    """Explode all multi-polygon geometries in a geodataframe into individual polygon geometries.

    Adds exploded polygons as rows at the end of the geodataframe and resets its index.
    """
    outdf = df[df.geom_type == 'Polygon']

    df_mp = df[df.geom_type == 'MultiPolygon']
    for idx, row in df_mp.iterrows():
        df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
        df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
        for i in range(len(row.geometry)):
            df_temp.loc[i, 'geometry'] = row.geometry[i]
        outdf = outdf.append(df_temp, ignore_index=True)

    outdf.reset_index(drop=True, inplace=True)
    return outdf


def keep_biggest_poly(df: GDF) -> GDF:
    """Replaces MultiPolygons with the biggest polygon contained in the MultiPolygon."""
    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()
    for idx in row_idxs_mp:
        mp = df.loc[idx].geometry
        poly_areas = [p.area for p in mp]
        max_area_poly = mp[poly_areas.index(max(poly_areas))]
        df.loc[idx, 'geometry'] = max_area_poly
    return df


def clip(df: GDF,
         clip_poly: Polygon,
         explode_mp_: bool = False,
         keep_biggest_poly_: bool = False,
         ) -> GDF:
    """Filter and clip geodataframe to clipping geometry.

    The clipping geometry needs to be in the same projection as the geodataframe.

    Args:
        df: input geodataframe
        clip_poly: Clipping polygon geometry, needs to be in the same crs as the input geodataframe.
        explode_mp_: Applies explode_mp function. Append dataframe rows for each polygon in potential
            multipolygons that were created by the intersection. Resets the dataframe index!
        keep_biggest_poly_: Applies keep_biggest_poly function. Replaces MultiPolygons with the biggest
        polygon contained in the MultiPolygon.

    Returns:
        Result geodataframe.
    """
    df = df[df.geometry.intersects(clip_poly)].copy()
    df.geometry = df.geometry.apply(lambda _p: _p.intersection(clip_poly))
    # df = gpd.overlay(df, clip_poly, how='intersection')  # Slower.

    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()

    if not row_idxs_mp:
        return df
    elif not explode_mp_ and (not keep_biggest_poly_):
        warnings.warn(f"Warning, intersection resulted in {len(row_idxs_mp)} split multipolygons. Use "
                      f"explode_mp_=True or keep_biggest_poly_=True.")
        return df
    elif explode_mp_ and keep_biggest_poly_:
        raise ValueError('You can only use one of "explode_mp_" or "keep_biggest_poly_"!')
    elif explode_mp_:
        return explode_mp(df)
    elif keep_biggest_poly_:
        return keep_biggest_poly(df)


def reclassify_col(df: Union[GDF, DF],
                   rcl_scheme: Dict,
                   col_classlabels: str= 'lcsub',
                   col_classids: str= 'lcsub_id',
                   drop_other_classes: bool=True
                   ) -> Union[GDF, DF]:
    """Reclassify class label and class ids in a dataframe column.

    # TODO: Simplify & make more efficient!
    Args:
        df: input geodataframe.
        rcl_scheme: Reclassification scheme, e.g. {'springcereal': [1,2,3], 'wintercereal': [10,11]}
        col_classlabels: column with class labels.
        col_classids: column with class ids.
        drop_other_classes: Drop classes that are not contained in the reclassification scheme.

    Returns:
        Result dataframe.
    """
    if drop_other_classes is True:
        classes_to_drop = [v for values in rcl_scheme.values() for v in values]
        df = df[df[col_classids].isin(classes_to_drop)].copy()

    rcl_dict = {}
    rcl_dict_id = {}
    for i, (key, value) in enumerate(rcl_scheme.items(), 1):
        for v in value:
            rcl_dict[v] = key
            rcl_dict_id[v] = i

    df[f'r_{col_classlabels}'] = df[col_classids].copy().map(rcl_dict)  # map name first, id second!
    df[f'r_{col_classids}'] = df[col_classids].map(rcl_dict_id)

    return df


reclass_legend = {
    'springcereal': [1, 2, 3, 4, 6, 7, 21, 55, 56, 210, 211, 212, 213, 214, 215, 224, 230, 234, 701, 702, 703, 704,
                     705],
    'wintercereal': [10, 11, 13, 14, 15, 16, 17, 22, 57, 220, 221, 222, 223, 235],
    'maize': [5, 216],
    'grassland': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121,
                  122, 123, 125, 126, 162, 170, 171, 172, 173, 174, 180, 182, 260, 261, 262, 263, 264, 266, 267,
                  268, 269, 270, 281, 282, 283, 284],
    'other': [23, 24, 25, 30, 31, 32, 35, 36, 40, 42, 51, 52, 53, 54, 55, 56, 57, 124, 160, 161, 280, 401, 402, 403,
              404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 418, 420, 421, 422, 423, 424, 429,
              430, 431, 432, 434, 440, 448, 449, 450, 487, 488, 489, 491, 493, 496, 497, 498, 499, 501, 502, 503,
              504, 505, 507, 509, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
              528, 529, 530, 531, 532, 533, 534, 536, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551,
              552, 553, 560, 561, 563, 570, 579]
    # drop other non-crop related classes (forest related, environment, recreation, other grass, permanent grass,
    # wasteland, ..)
    }


def reduce_precision(ingeo: Union[Polygon, GDF], precision: int=3) -> Union[Polygon, GDF]:
    """Reduces the number of after comma decimals of a shapely Polygon or geodataframe geometries.

    GeoJSON specification recommends 6 decimal places for latitude and longitude which equates to roughly 10cm of
    precision (https://github.com/perrygeo/geojson-precision).

    Args:
        ingeo: input geodataframe or shapely Polygon.
        precision: number of after comma values that should remain.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _reduce_precision(poly: Polygon, precision: int) -> Polygon:
        geojson = shapely.geometry.mapping(poly)
        geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision)
        poly = shapely.geometry.shape(geojson)
        if not poly.is_valid:  # Too low precision can potentially lead to invalid polygons due to line overlap effects.
            poly = poly.buffer(0)
        return poly

    if isinstance(ingeo, Polygon):
        return _reduce_precision(poly=ingeo, precision=precision)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _reduce_precision(poly=_p, precision=precision))
        return ingeo


def to_pixelcoords(ingeo: Union[Polygon, GDF],
                   reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
                   scale: bool=False,
                   nrows: int=None,
                   ncols: int=None
                   ) -> Union[Polygon, GDF]:
    """Converts projected polygon coordinates to pixel coordinates of an image array.

    Subtracts point of origin, scales to pixelcoordinates.

    Input:
        ingeo: input geodataframe or shapely Polygon.
        reference_bounds:  Bounding box object or tuple of reference (e.g. image chip) in format (left, bottom,
            right, top)
        scale: Scale the polygons to the image size/resolution. Requires image array nrows and ncols parameters.
        nrows: image array nrows, required for scale.
        ncols: image array ncols, required for scale.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _to_pixelcoords(poly: Polygon, reference_bounds, scale, nrows, ncols):
        try:
            minx, miny, maxx, maxy = reference_bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'reference_bounds argument is of type {type(reference_bounds)}, needs to be a tuple or rasterio bounding box '
                f'instance. Can be delineated from transform, nrows, ncols via rasterio.transform.reference_bounds')

        # Subtract point of origin of image bbox.
        x_coords, y_coords = poly.exterior.coords.xy
        p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            if ncols is None or nrows is None:
                raise ValueError('ncols and nrows required for scale')
            x_scaler = ncols / w_poly
            y_scaler = nrows / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    if isinstance(ingeo, Polygon):
        return _to_pixelcoords(poly=ingeo, reference_bounds=reference_bounds, scale=scale, nrows=nrows, ncols=ncols)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, reference_bounds=reference_bounds,
                                                                         scale=scale, nrows=nrows, ncols=ncols))
        return ingeo


def invert_y_axis(ingeo: Union[Polygon, GDF],
                  reference_height: int
                  ) -> Union[Polygon, GDF]:
    """Invert y-axis of polygon or geodataframe geometries in reference to a bounding box e.g. of an image chip.

    Usage e.g. for COCOJson format.

    Args:
        ingeo: Input Polygon or geodataframe.
        reference_height: Height (in coordinates or rows) of reference object (polygon or image, e.g. image chip.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _invert_y_axis(poly: Polygon=ingeo, reference_height=reference_height):
        x_coords, y_coords = poly.exterior.coords.xy
        p_inverted_y_axis = shapely.geometry.Polygon([[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        return p_inverted_y_axis

    if isinstance(ingeo, Polygon):
        return _invert_y_axis(poly=ingeo, reference_height=reference_height)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _invert_y_axis(poly=_p, reference_height=reference_height))
        return ingeo


def cut_chip_geometries(vector_df, raster_width, raster_height, raster_transform, chip_width=128, chip_height=128, first_n_chips=None):
    """Workflow to cut a vector geodataframe to chip geometries.

    Filters small polygons and skips empty chips.

    Args:
        vector_df: Geodataframe containing the geometries to be cut to chip geometries.
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        first_n_chips: Only processes the first n image chips, used for debugging.

    Returns: Dictionary containing the final chip_df, chip_window, chip_transform, chip_poly objects.
    """

    generator_window_bounds = utils.img.get_chip_windows(raster_width=raster_width,
                                                         raster_height=raster_height,
                                                         raster_transform=raster_transform,
                                                         chip_width=chip_width,
                                                         chip_height=chip_height,
                                                         skip_partial_chips=True)

    all_chip_dfs = {}
    for i, (chip_window, chip_transform, chip_poly) in enumerate(tqdm(generator_window_bounds)):
        if i >= first_n_chips:
            break

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
            chip_df = chip_df.pipe(invert_y_axis, reference_height=chip_height)
        else:
            continue

        chip_name = f'COCO_train2016_000000{100000+i}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'
        all_chip_dfs[chip_name] = {'chip_df': chip_df,
                                   'chip_window': chip_window,
                                   'chip_transform': chip_transform,
                                   'chip_poly': chip_poly}
    return all_chip_dfs