""" Utils """

import os

import buteo as beo
import numpy as np


def clone_folder_structure(dest_folder, folder_structure):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest_structure = os.path.join(dest_folder, folder_structure)
    if not os.path.exists(dest_structure):
        os.makedirs(dest_structure)


def encode_latitude(lat):
    """ Latitude goes from -90 to 90 """

    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = (np.sin(2 * np.pi * (lat_adj / lat_max)) + 1) / 2.0
    encoded_cos = (np.cos(2 * np.pi * (lat_adj / lat_max)) + 1) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


def encode_longitude(lng):
    """ Longitude goes from -180 to 180 """

    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = (np.sin(2 * np.pi * (lng_adj / lng_max)) + 1) / 2.0
    encoded_cos = (np.cos(2 * np.pi * (lng_adj / lng_max)) + 1) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


def read_raster(raster, filled=True, fill_value=0, pixel_offsets=None, channel_last=False, cast=np.float32):
    return beo.raster_to_array(
        beo.raster_open(raster, writeable=False),
        filled=filled,
        fill_value=fill_value,
        pixel_offsets=pixel_offsets,
        channel_last=channel_last,
        cast=cast,
        )


def read_latlng_bounds(raster):
    metadata = beo.raster_to_metadata(beo.raster_open(raster, writeable=False))
    bounds_wgs84 = metadata["bbox_latlng"]
    lat, lng = metadata["centroid_latlng"]

    return lat, lng, bounds_wgs84


def create_reference(raster, pixel_offsets):
    return beo.array_to_raster(
        beo.raster_to_array(beo.raster_open(raster, writeable=False), pixel_offsets=pixel_offsets),
        reference=beo.raster_open(raster, writeable=False),
        pixel_offsets=pixel_offsets,
    )


def clip_and_read(raster, clip_geom, filled=True, fill_value=0, channel_last=False, cast=None):
    clipped = beo.raster_clip(beo.raster_open(raster, writeable=True), clip_geom=clip_geom)
    array = read_raster(clipped, filled=filled, fill_value=fill_value, channel_last=channel_last, cast=cast)
    beo.delete_dataset_if_in_memory(clipped),

    return array


def is_1_within_2(bbox1, bbox2):
    return beo.utils_bbox._check_bboxes_within(bbox1, bbox2)


def delete_memory_layer(layer):
    beo.delete_dataset_if_in_memory(layer)


def read_globals(label_path):
    paths = {
        "water": os.path.join(label_path, "water_4326.tif"),
        "terrain": os.path.join(label_path, "terrain_4326.tif"),
        "climate": os.path.join(label_path, "climate_4326.tif"),
        "landcover": os.path.join(label_path, "landcover_4326.tif"),
        "degurba": os.path.join(label_path, "degurba_4326.tif"),
        "buildings": os.path.join(label_path, "build_4326.tif"),
    }

    bboxes = {
        "water": beo.get_bbox_from_dataset(paths["water"]),
        "terrain": beo.get_bbox_from_dataset(paths["terrain"]),
        "climate": beo.get_bbox_from_dataset(paths["climate"]),
        "landcover": beo.get_bbox_from_dataset(paths["landcover"]),
        "degurba": beo.get_bbox_from_dataset(paths["degurba"]),
        "buildings": beo.get_bbox_from_dataset(paths["buildings"]),
    }

    return paths, bboxes
