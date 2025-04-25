import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import utils


class MajorTOM(torch.utils.data.Dataset):
    def __init__(
            self,
            MajorTOM_dataset_path="path/to/majortom/folder",
            MajorTOM_pickle_path="path/to/majortom/folder/pickle.pkl",
            label_folder_path="path/to/data_static",
            patch_size=128,
            read_static_to_ram=False,
            device="cuda",
            transform=None,
    ):
        self.dataset_path = os.path.join(MajorTOM_dataset_path, "Core-S2L2A/L2A")
        self.pickle_path = MajorTOM_pickle_path
        self.label_path = label_folder_path
        self.patch_size = patch_size
        self.read_static_to_ram = read_static_to_ram
        self.device = device
        self.transform = transform

        if self.pickle_path:
            self.dataset = pd.read_pickle(self.pickle_path)
        else:
            raise Exception("No {MajorTOM_pickle_path} specified!")

        self.global_static_labels, self.global_static_bounds = utils.read_globals(self.label_path)

        self.band_scales = {
            "10m": ["B02", "B03", "B04", "B08", "Cloud_mask"],
            "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60m": ["B01", "B09"],
        }

        self.band_sizes = {
            "10m": (1068, 1068),
            "20m": (534, 534),
            "60m": (178, 178),
        }

        self.landcover_map = np.zeros(101, dtype=np.uint8)
        self.landcover_map[0] = 0  # nodata
        self.landcover_map[10] = 1  # trees
        self.landcover_map[20] = 2  # shrubs
        self.landcover_map[30] = 3  # grass
        self.landcover_map[40] = 4  # crops
        self.landcover_map[50] = 5  # built
        self.landcover_map[60] = 6  # bare
        self.landcover_map[70] = 7  # snow
        self.landcover_map[80] = 8  # water
        self.landcover_map[90] = 9  # wetland
        self.landcover_map[95] = 10  # mangrove
        self.landcover_map[100] = 11  # moss/lichen

    def get_random_offset(self):
        pixel_20m = self.patch_size // 2
        height = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2
        width = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2

        return width, height, self.patch_size, self.patch_size

    def compose_paths(self, row):
        """
        :param row: contains row, column and name of the specific data
        :return: list of file paths
        """

        data_path = os.path.join(self.dataset_path, row['row'], f"{row['row']}_{row['column']}", row['L2A_name'])

        paths = {
            "B01": os.path.join(data_path, "B01.tif"),
            "B02": os.path.join(data_path, "B02.tif"),
            "B03": os.path.join(data_path, "B03.tif"),
            "B04": os.path.join(data_path, "B04.tif"),
            "B05": os.path.join(data_path, "B05.tif"),
            "B06": os.path.join(data_path, "B06.tif"),
            "B07": os.path.join(data_path, "B07.tif"),
            "B08": os.path.join(data_path, "B08.tif"),
            "B8A": os.path.join(data_path, "B8A.tif"),
            "B09": os.path.join(data_path, "B09.tif"),
            "B11": os.path.join(data_path, "B11.tif"),
            "B12": os.path.join(data_path, "B12.tif"),
            "Cloud_mask": os.path.join(data_path, "cloud_mask.tif"),
            "Thumbnail": os.path.join(data_path, "thumbnail.png")
        }

        return paths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        paths = self.compose_paths(row)

        bbox = self.get_random_offset()

        try:
            reference_raster = utils.create_reference(paths["B05"], pixel_offsets=[v // 2 for v in bbox])
            lat, lng, bounds_wgs84 = utils.read_latlng_bounds(reference_raster)

            bands = {}
            for band in self.band_scales["10m"]:
                bands[band] = utils.read_raster(paths[band], pixel_offsets=bbox)

            for band in self.band_scales["20m"]:
                bands[band] = utils.read_raster(paths[band], pixel_offsets=[v // 2 for v in bbox])

            for band in bands:
                bands[band] = torch.tensor(bands[band], dtype=torch.float32, device=self.device) / 10000.0

                if band in self.band_scales["20m"]:
                    bands[band] = F.interpolate(bands[band].unsqueeze(0), scale_factor=2, mode="bilinear").squeeze(0)

            clouds = torch.tensor(
                [
                    (bands["Cloud_mask"] == 0).sum() / bands["Cloud_mask"].numel(),
                    (bands["Cloud_mask"] == 1).sum() / bands["Cloud_mask"].numel(),
                    (bands["Cloud_mask"] == 2).sum() / bands["Cloud_mask"].numel(),
                    (bands["Cloud_mask"] == 3).sum() / bands["Cloud_mask"].numel(),
                ], dtype=torch.float32, device=self.device
            )
            clouds_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            coords = torch.tensor(
                np.concatenate([utils.encode_latitude(lat), utils.encode_longitude(lng)]),
                dtype=torch.float32,
                device=self.device,
            )
            coords_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            if utils.is_1_within_2(bounds_wgs84, self.global_static_bounds["water"]):
                water_array = utils.clip_and_read(self.global_static_labels["water"], clip_geom=reference_raster)
                water_value = water_array.sum() / water_array.size
                water = torch.tensor(water_value, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                water = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if utils.is_1_within_2(bounds_wgs84, self.global_static_bounds["buildings"]):
                buildings_array = utils.clip_and_read(
                    self.global_static_labels["buildings"],
                    clip_geom=reference_raster
                )
                buildings_value = (buildings_array.sum() / buildings_array.size) / 100.0
                buildings = torch.tensor(buildings_value, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                buildings = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if utils.is_1_within_2(bounds_wgs84, self.global_static_bounds["landcover"]):
                landcover_array = utils.clip_and_read(
                    self.global_static_labels["landcover"],
                    clip_geom=reference_raster
                ).flatten()
                landcover_non_zero = np.count_nonzero(landcover_array)
                map_landcover = np.vectorize(lambda x: self.landcover_map[x])

                if landcover_non_zero == 0.0:
                    landcover = torch.tensor(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32, device=self.device
                    )
                    landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    mapped = map_landcover(landcover_array)
                    landcover_values = np.bincount(mapped, minlength=12)[1:] / landcover_non_zero
                    landcover = torch.tensor(landcover_values, dtype=torch.float32, device=self.device)
                    landcover_weight = torch.tensor(
                        landcover_non_zero / landcover_array.size, dtype=torch.float32,
                        device=self.device
                    )

            else:
                landcover = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                    device=self.device
                )
                landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            utils.delete_memory_layer(reference_raster)

            x = torch.cat(
                [
                    bands["B02"],
                    bands["B03"],
                    bands["B04"],
                    bands["B08"],
                    bands["B05"],
                    bands["B06"],
                    bands["B07"],
                    bands["B8A"],
                    bands["B11"],
                    bands["B12"],
                ], dim=0
            )

            label = {
                "coords": coords, "coords_weight": coords_weight,
                "clouds": clouds, "cloud_weight": clouds_weight,
                "water": water, "water_weight": water_weight,
                "buildings": buildings, "buildings_weight": buildings_weight,
                "landcover": landcover, "landcover_weight": landcover_weight,
            }

            if self.transform is not None:
                x = self.transform(x)

        except:
            new_idx = 0 if idx + 1 == len(self.dataset) else idx + 1
            return self.__getitem__(new_idx)

        return x, label