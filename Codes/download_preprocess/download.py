# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import ee
import sys
import logging
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Optional
from rasterio.transform import from_bounds
from pycropwat import EffectivePrecipitation

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utils.raster_ops import clip_resample_reproject_raster

# ***************************************** earth engine authentication *************************************

# ee.Authenticate()

# **********************************************************************************************************************

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

# reference values - shapefiles - rasters
no_data_value = -9999
res_2km = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS.shp'
WestUS_raster = PROJECT_ROOT / 'Data_main/ref_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_for_2km = PROJECT_ROOT / 'Data_main/ref_rasters/GEE_merging_refraster_larger_grids.tif'
gee_grid_shape_large = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_gee_grid_large.shp'

# Maximum pixels per tile for GEE sampleRectangle (conservative limit)
MAX_PIXELS_PER_TILE = 65536  # 256 x 256


class GEE_download:
    def __init__(self, ee_project: str = 'ee-fahim'):
        self.ee_project = ee_project

    def get_gee_dict(self, data_name):

        # Initialize earth engine
        ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

        gee_data_dict = {
            'GRIDMET_Precip': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_RET': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_Tmax': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_maxRH': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_minRH': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_windVel': 'IDAHO_EPSCOR/GRIDMET',  # at 10m
            'GRIDMET_shortRad': 'IDAHO_EPSCOR/GRIDMET',
            'GRIDMET_vpd': 'IDAHO_EPSCOR/GRIDMET',
            'DAYMET_sunHr': 'NASA/ORNL/DAYMET_V4',
            'PRISM_Precip': 'projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY',
            'PRISM_Tmax': 'projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY',
            'PRISM_Tmean': 'projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY',
            'USDA_CDL': 'USDA/NASS/CDL',
            'Field_capacity': 'OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01',
            'Bulk_density': 'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02',
            'Sand_content': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
            'Clay_content': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
            'DEM': 'USGS/SRTMGL1_003',
            'Tree_cover': 'NASA/MEASURES/GFCC/TC/v3',
            'spi': 'GRIDMET/DROUGHT',  # Standardized Precipitation Index (precipitation anomalies)
            'spei': 'GRIDMET/DROUGHT',
            # Standardized Precipitation Evapotranspiration Index (temperature-driven drought-water balance)
            'eddi': 'GRIDMET/DROUGHT',  # Evaporative Drought Demand Index (atmospheric drying demand)
            'LANID_1997_2017': 'projects/openet/irrigated_area/LANID',  # GEE imagecollection
            'LANID_2018_2025': 'projects/routinelanid/assets/LANID/LANID2018-2025',  # GEE image
        }

        gee_band_dict = {
            'GRIDMET_Precip': 'pr',  # daily total, unit in mm
            'GRIDMET_RET': 'etr',
            'GRIDMET_Tmax': 'tmmx',  # unit in K
            'GRIDMET_maxRH': 'rmax',
            'GRIDMET_minRH': 'rmin',
            'GRIDMET_windVel': 'vs',
            'GRIDMET_shortRad': 'srad',
            'GRIDMET_vpd': 'vpd',
            'DAYMET_sunHr': 'dayl',
            'PRISM_Precip': 'ppt',
            'PRISM_Tmax': 'tmax',
            'PRISM_Tmean': 'tmean',
            'USDA_CDL': 'cropland',
            'Field_capacity': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
            'Bulk_density': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
            'Sand_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
            'Clay_content': ['b0', 'b10', 'b30', 'b60', 'b100', 'b200'],
            'DEM': 'elevation',
            'Tree_cover': 'tree_canopy_cover',
            'spi': 'spi1y',
            'spei': 'spi1y',
            'eddi': 'eddi1y',
            'LANID_1997_2017': 'irr_land',  # have to select year by filterdate
            'LANID_2018_2025': None,  # called by last 2 digits of year (e.g., '18' for 2018) in GEE image
        }

        gee_scale_dict = {
            'GRIDMET_Precip': 1,
            'GRIDMET_RET': 1,
            'GRIDMET_Tmax': 1,
            'GRIDMET_maxRH': 1,
            'GRIDMET_minRH': 1,
            'GRIDMET_windVel': 1,
            'GRIDMET_shortRad': 1,
            'GRIDMET_vpd': 1,
            'DAYMET_sunHr': 1,
            'PRISM_Precip': 1,
            'PRISM_Tmax': 1,
            'PRISM_Tmean': 1,
            'USDA_CDL': 1,
            'Field_capacity': 1,
            'Bulk_density': 1,
            'Organic_carbon_content': 1,
            'Sand_content': 1,
            'Clay_content': 1,
            'DEM': 1,
            'Tree_cover': 1,
            'spi': 1,
            'spei': 1,
            'eddi': 1,
            'LANID_1997_2017': 1,
            'LANID_2018_2025': 1
        }

        aggregation_dict = {
            'GRIDMET_Precip': ee.Reducer.mean(),
            'GRIDMET_RET': ee.Reducer.mean(),
            'GRIDMET_Tmax': ee.Reducer.mean(),
            'GRIDMET_maxRH': ee.Reducer.mean(),
            'GRIDMET_minRH': ee.Reducer.mean(),
            'GRIDMET_windVel': ee.Reducer.mean(),
            'GRIDMET_shortRad': ee.Reducer.mean(),
            'GRIDMET_vpd': ee.Reducer.mean(),
            'DAYMET_sunHr': ee.Reducer.mean(),
            'PRISM_Precip': ee.Reducer.mean(),
            'PRISM_Tmax': ee.Reducer.mean(),
            'PRISM_Tmean': ee.Reducer.mean(),
            'USDA_CDL': ee.Reducer.first(),
            'Field_capacity': ee.Reducer.mean(),
            'Bulk_density': ee.Reducer.mean(),
            'Sand_content': ee.Reducer.mean(),
            'Clay_content': ee.Reducer.mean(),
            'DEM': None,
            'Tree_cover': ee.Reducer.mean(),
            'spi': ee.Reducer.mean(),
            'spei': ee.Reducer.mean(),
            'eddi': ee.Reducer.mean(),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None
        }

        # # Note on start date and end date dictionaries
        # The start and end dates have been set based on what duration of data can be downloaded.
        # They may not exactly match with the data availability in GEE
        # In most cases the end date is shifted a month later to cover the end month's data

        month_start_date_dict = {
            'GRIDMET_Precip': datetime(1979, 1, 1),
            'GRIDMET_RET': datetime(1979, 1, 1),
            'GRIDMET_Tmax': datetime(1979, 1, 1),
            'GRIDMET_maxRH': datetime(1979, 1, 1),
            'GRIDMET_minRH': datetime(1979, 1, 1),
            'GRIDMET_windVel': datetime(1979, 1, 1),
            'GRIDMET_shortRad': datetime(1979, 1, 1),
            'GRIDMET_vpd': datetime(1979, 1, 1),
            'DAYMET_sunHr': datetime(1980, 1, 1),
            'PRISM_Precip': datetime(1895, 1, 1),
            'PRISM_Tmax': datetime(1895, 1, 1),
            'PRISM_Tmean': datetime(1895, 1, 1),
            'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
            'Field_capacity': None,
            'Bulk_density': None,
            'Sand_content': None,
            'Clay_content': None,
            'DEM': None,
            'Tree_cover': datetime(2000, 1, 1),
            'spi': datetime(1980, 1, 5),
            'spei': datetime(1980, 1, 5),
            'eddi': datetime(1980, 1, 5),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None
        }

        month_end_date_dict = {
            'GRIDMET_Precip': datetime(2023, 9, 15),
            'GRIDMET_RET': datetime(2024, 1, 1),
            'GRIDMET_Tmax': datetime(2024, 1, 1),
            'GRIDMET_maxRH': datetime(2024, 1, 1),
            'GRIDMET_minRH': datetime(2024, 1, 1),
            'GRIDMET_windVel': datetime(2024, 1, 1),
            'GRIDMET_shortRad': datetime(2024, 1, 1),
            'GRIDMET_vpd': datetime(2024, 1, 1),
            'PRISM_Precip': datetime(2025, 1, 1),
            'PRISM_Tmax': datetime(2025, 1, 1),
            'PRISM_Tmean': datetime(2025, 1, 1),
            'DAYMET_sunHr': datetime(2025, 1, 1),
            'USDA_CDL': datetime(2022, 1, 1),
            'Field_capacity': None,
            'Bulk_density': None,
            'Sand_content': None,
            'Clay_content': None,
            'DEM': None,
            'Tree_cover': datetime(2015, 1, 1),
            'spi': datetime(2024, 12, 31),
            'spei': datetime(2024, 12, 31),
            'eddi': datetime(2024, 12, 31),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None
        }

        year_start_date_dict = {
            'GRIDMET_Precip': datetime(1979, 1, 1),
            'GRIDMET_RET': datetime(1979, 1, 1),
            'GRIDMET_Tmax': datetime(1979, 1, 1),
            'GRIDMET_maxRH': datetime(1979, 1, 1),
            'GRIDMET_minRH': datetime(1979, 1, 1),
            'GRIDMET_windVel': datetime(1979, 1, 1),
            'GRIDMET_shortRad': datetime(1979, 1, 1),
            'GRIDMET_vpd': datetime(1979, 1, 1),
            'PRISM_Precip': datetime(1895, 1, 1),
            'PRISM_Tmax': datetime(1895, 1, 1),
            'PRISM_Tmean': datetime(1895, 1, 1),
            'DAYMET_sunHr': datetime(1980, 1, 1),
            'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
            'Field_capacity': None,
            'Bulk_density': None,
            'Sand_content': None,
            'Clay_content': None,
            'DEM': None,
            'Tree_cover': datetime(2000, 1, 1),
            'spi': datetime(1980, 1, 5),
            'spei': datetime(1980, 1, 5),
            'eddi': datetime(1980, 1, 5),
            'LANID_1997_2017': datetime(1998, 1, 1),
            'LANID_2018_2025': datetime(2018, 1, 1)
        }

        year_end_date_dict = {
            'GRIDMET_Precip': datetime(2024, 1, 1),
            'GRIDMET_RET': datetime(2024, 12, 1),
            'GRIDMET_Tmax': datetime(2024, 12, 1),
            'GRIDMET_maxRH': datetime(2024, 1, 1),
            'GRIDMET_minRH': datetime(2024, 1, 1),
            'GRIDMET_windVel': datetime(2024, 1, 1),
            'GRIDMET_shortRad': datetime(2024, 1, 1),
            'GRIDMET_vpd': datetime(2024, 12, 1),
            'PRISM_Precip': datetime(2025, 1, 1),
            'PRISM_Tmax': datetime(2025, 1, 1),
            'PRISM_Tmean': datetime(2025, 1, 1),
            'DAYMET_sunHr': datetime(2024, 1, 1),
            'USDA_CDL': datetime(2022, 1, 1),
            'Field_capacity': None,
            'Bulk_density': None,
            'Sand_content': None,
            'Clay_content': None,
            'DEM': None,
            'Tree_cover': datetime(2015, 1, 1),
            'spi': datetime(2024, 12, 31),
            'spei': datetime(2024, 12, 31),
            'eddi': datetime(2024, 12, 31),
            'LANID_1997_2017': datetime(2017, 12, 31),
            'LANID_2018_2025': datetime(2025, 12, 31)
        }

        return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[
            data_name], \
            month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
            year_end_date_dict[data_name]


    @staticmethod
    def __estimate_pixel_count(bounds_coords: np.array,
                               scale_meters: int
                               ) -> int:
        """
        Estimate the number of pixels required in the given bounds and resolution.

        :param bounds_coords: Numpy Array or list
                              Total bounds [minx, miny, maxx, maxy].
                              Generally comes from geopandas shapefile.total_bounds.

        :param scale_meters: Int
                             Target resolution in meters.

        :return: Int
                 Estimated number of pixels.
        """
        min_lon = bounds_coords[0]
        min_lat = bounds_coords[1]
        max_lon = bounds_coords[2]
        max_lat = bounds_coords[3]

        # approximate width and height in meters (at mid-latitude)
        mid_lat = (min_lat + max_lat) / 2
        lat_meters_per_degree = 111320  # meters per degree latitude
        lon_meters_per_degree = 111320 * np.cos(np.radians(mid_lat))

        width_meters = (max_lon - min_lon) * lon_meters_per_degree
        height_meters = (max_lat - min_lat) * lat_meters_per_degree

        n_cols = int(np.ceil(width_meters / scale_meters))
        n_rows = int(np.ceil(height_meters / scale_meters))

        return n_cols * n_rows


    @staticmethod
    def __create_tile_grid(bounds_coords: np.array,
                           scale_meters: int
                           ) -> List[ee.Geometry]:
        """
        Create grid tiles that covers the bounding box for chunked data processing for download.

             :param bounds_coords: Numpy Array or list
                              Total bounds [minx, miny, maxx, maxy].
                              Generally comes from geopandas shapefile.total_bounds.
        :param scale_meters: Int
                             Target resolution in meters.

        :return: List
                 A list of ee.Geometry.Rectangle tiles.
        """
        min_lon = bounds_coords[0]
        min_lat = bounds_coords[1]
        max_lon = bounds_coords[2]
        max_lat = bounds_coords[3]

        # Calculate tile size in degrees based on MAX_PIXELS_PER_TILE
        tile_pixel_per_side = int(np.sqrt(MAX_PIXELS_PER_TILE))  # e.g., 256 pixels per side

        # approximate width and height in meters (at mid-latitude)
        mid_lat = (min_lat + max_lat) / 2
        lat_meters_per_degree = 111320  # meters per degree latitude
        lon_meters_per_degree = 111328 * np.cos(np.radians(mid_lat))

        # Tile size in degrees
        tile_width_in_degree = (tile_pixel_per_side * scale_meters) / lon_meters_per_degree
        tile_height_in_degree = (tile_pixel_per_side * scale_meters) / lat_meters_per_degree

        tiles = []
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            while lon < max_lon:
                tile_max_lat = min(lat + tile_height_in_degree, max_lat)
                tile_max_lon = min(lon + tile_width_in_degree, max_lon)

                # creating tile in earth engine geometry
                tile = ee.Geometry.Rectangle([lon, lat, tile_max_lon, tile_max_lat])
                tiles.append(tile)

                lon += tile_width_in_degree
            lat += tile_height_in_degree

        # logger.info(f'created len{tiles} tiles for download')

        return tiles


    @staticmethod
    def __download_single_tile(img: ee.Image,
                               tile: ee.Geometry,
                               band_name: str,
                               default_nodata: float = 0,
                               ) -> Optional[Tuple[np.ndarray, List]]:
        """
        Download a single image time from GEE. The tile will not be physically downloaded, rather stay in
        background to be styched together with the other tiles.

        :param img: ee.Image
                    Earth engine image to download.

        :param tile: ee.Geometry
                    Earth engine geometry tile.

        :param band_name: str
                    Name of band to extract from the image.

        :param default_nodata: float
                    No data value in the data. Default set to 0.


        :return: tuple or None
                 Tuple of (array, coordinates) of the image or None if download fails.
        """
        try:
            arr = img.sampleRectangle(
                region=tile,
                defaultValue=default_nodata,
            ).get(band_name).getInfo()

            if arr is None:
                return None

            arr = np.array(arr, dtype=np.float32)
            coords = tile.getInfo()['coordinates'][0]

            return arr, coords

        except Exception as e:
            logger.warning(f'Failed to download tile: {e}')
            # logger.exception("Tile download crashed")
            return None


    def __download_image_chunked(self,
                                 img: ee.Image,
                                 bounds_coords: np.array,
                                 scale_meters: int,
                                 band_name: str,
                                 download_dir: str,
                                 year: int,
                                 month: int = None,
                                 default_nodata: float = 0,
                                 data_name: str = 'data',
                                 n_workers: int = 5,
                                 clip_resample_to_target_raster: bool = True,
                                 clip_shapefile: str = WestUS_shape,
                                 ref_raster: str = WestUS_raster,
                                 clip_resample_resolution: float = res_2km) -> np.ndarray:
        """
        Download a large Earth Engine image by splitting the requested bounding box
        into smaller tiles and processing them in parallel.

        :param img: ee.Image
            Earth Engine image object to download.

        :param bounds_coords: np.ndarray or list
            Bounding box coordinates [minx, miny, maxx, maxy].

        :param scale_meters: int
            Spatial resolution in meters.

        :param band_name: str
            Band name to extract from the image.

        :param download_dir: str
            Directory where the output raster will be saved.

        :param year: int
            Year of the dataset being downloaded.

        :param month: int, optional
            Month of the dataset being downloaded.

        :param default_nodata: float
            Value used for missing data. Default is 0.

        :param data_name: str
            Name of the dataset (used for naming output files).

        :param n_workers: int
            Number of parallel workers used for tile downloading.

        :param clip_resample_to_target_raster: bool
            If True, clip and resample output to reference raster. Set to False to skip this process and keep the
            original download intact.

        :param clip_shapefile: str
            Filepath of shapefile to clip and resample downloaded and merged raster.

        :param ref_raster: str
            Path to reference raster for clipping/resampling. Can be set to 'None' if
            'clip_resample_to_ref_raster = False'.

        :param clip_resample_resolution: float
            Resolution of the final data. Used in the process of clip & resample to make all downloaded data aligned
            with a reference raster. Can be set to 'None' if 'clip_resample_to_ref_raster = False'.

        :return: None
        """

        try:
            # create tiles
            tiles = self.__create_tile_grid(bounds_coords, scale_meters)
            logger.info(f'Downloading {data_name} in {len(tiles)} tiles...')

            # Creating delayed tasks for dak parallel processing
            tasks = [delayed(self.__download_single_tile)
                     (img, tile, band_name, default_nodata)
                     for tile in tiles]

            # Execute in parallel with progress bar
            with ProgressBar():
                results = compute(*tasks, num_workers=n_workers)

            tile_arrays = []
            tile_coords = []

            for result in results:
                if result is not None:
                    arr, coords = result
                    tile_arrays.append(arr)
                    tile_coords.append(coords)

            if len(tile_arrays) == 0:
                raise ValueError('Download failed. No tiles downloaded')


            # Mosaic tiles together
            min_lon = bounds_coords[0]
            min_lat = bounds_coords[1]
            max_lon = bounds_coords[2]
            max_lat = bounds_coords[3]

            # calculate output dimensions
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon

            scale_deg = scale_meters / 111320  # approximate meters per degree
            out_rows = int(lat_range / scale_deg)
            out_cols = int(lon_range / scale_deg)

            # create an empty output array
            output_arr = np.full((out_rows, out_cols), np.nan, dtype=np.float32)

            # place each tile into the output array
            for arr, coords in zip(tile_arrays, tile_coords):
                tile_min_lon = min(c[0] for c in coords)
                tile_max_lat = max(c[1] for c in coords)

                # calculate pixel indices
                # pixel indices are used to decide where the tile should be placed in the big output array
                col_start = int((tile_min_lon - min_lon) / scale_deg)
                row_start = int((max_lat - tile_max_lat) / scale_deg)

                # place data
                rows = min(arr.shape[0], out_rows - row_start)
                cols = min(arr.shape[1], out_cols - col_start)

                if rows > 0 and cols > 0:
                    output_arr[row_start:row_start+rows, col_start:col_start+cols] = arr[:rows, :cols]

            # Fill NaN with default
            output_arr = np.nan_to_num(output_arr, nan=default_nodata)

            # save merged array
            merged_download_dir = Path(download_dir) / 'raw_download'
            merged_download_dir.mkdir(parents=True, exist_ok=True)

            downloaded_raster = (
                self.__save_raster_from_arr_coords(bounds_coords, output_arr,
                                                   merged_download_dir,
                                                   data_name, year, month))

            # clip and resample the downloaded array to the reference raster
            if clip_resample_to_target_raster:
                clip_resample_reproject_raster(input_raster=downloaded_raster,
                                               input_shape=clip_shapefile,
                                               output_raster_dir=download_dir,
                                               clip_and_resample=True,
                                               ref_raster=ref_raster,
                                               resolution=clip_resample_resolution)

            logger.info(f"Successfully mosaicked - clipped - resampled {len(tile_arrays)} {data_name} tiles.")
            logger.info('---------------------------------------------------------------------------------\n')

        except Exception as e:
            logger.warning(f"{data_name} chunked download failed: {e}.")

            raise

    @staticmethod
    def __save_raster_from_arr_coords(download_bounds: np.ndarray or list,
                                      arr: np.ndarray,
                                      download_dir: str,
                                      data_name: str,
                                      year: int,
                                      month: int = None) -> Path:

        # save the downloaded array
        transform = from_bounds(
            download_bounds[0],
            download_bounds[1],
            download_bounds[2],
            download_bounds[3],
            arr.shape[1],  # width
            arr.shape[0]   # height
        )

        suffix = f'{data_name}_{year}_{month}.tif' if month is not None else f'{data_name}_{year}.tif'
        output_file = Path(download_dir) / suffix


        with rasterio.open(
                output_file,
                'w',
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=arr.dtype,
                crs="EPSG:4326",
                transform=transform,
                nodata=0
        ) as dst:
            dst.write(arr, 1)

        return output_file

    def download_gee_data_monthly(self,
                                  data_name: str,
                                  main_download_dir: str,
                                  input_shape_for_data_download: str,
                                  year_list: list,
                                  month_range: tuple,
                                  ee_reducer: ee.Reducer,
                                  scale_meters: int = 2200,
                                  clip_resample_to_target_raster: bool = True,
                                  clip_resample_resolution: float = res_2km,
                                  ref_raster: str = WestUS_raster,
                                  use_cpu_while_multidownloading: int = 15
                                  ) -> None:
        """
        Download monthly data from Google Earth Engine (GEE) for the specified
        dataset, years, and month range.

        *** You can only download dataset declared in get_gee_dict()
        *** To download other datasets, please add the assets, band names, and start-end of data in get_gee_dict()
        *** Change 'ee_reducer' to your desired reducer if you don't prefer the default reducer from get_gee_dict()
        *** If the bounding region exceeds the Earth Engine pixel limit (MAX_PIXELS_PER_TILE), the function
            automatically switches to chunked parallel downloading.

        :param data_name : str
            Name of the dataset to download. Must be a valid key supported
            by `get_gee_dict()` (e.g., 'GRIDMET_Precip', 'GRIDMET_RET',
            'PRISM_Precip', etc.).

        :param main_download_dir : str or Path
            Directory where downloaded rasters will be saved.

        :param input_shape_for_data_download : str or Path
            Path to a shapefile defining the spatial extent to download.

        :param year_list : list[int]
            List of years to download.

        :param month_range : tuple[int, int]
            Inclusive range of months to download (e.g., (1, 12)).

        :param ee_reducer: ee.Reducer
            Use a desired reducer (e.g., ee.Reducer.mean()).
            Default set to None to use default reducer from get_gee_dict().

        :param scale_meters : int, optional
            Target spatial resolution in meters. Default is 2200.

        :param clip_resample_to_target_raster: bool
            If True, clip and resample output to reference raster. Set to False to skip this process and keep the
            original download intact.

        :param clip_resample_resolution float
            Resolution of the final data. Used in the process of clip & resample to make all downloaded data aligned
            with a reference raster.

        :param ref_raster : str or Path, optional
            Reference raster used for clipping and resampling.

        :param use_cpu_while_multidownloading : int, optional
            Number of parallel workers used during tiled download.
            Default is 15.

        :return None.
        """

        # Initialize earth engine
        ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

        download_dir = Path(main_download_dir) / data_name

        download_dir.mkdir(parents=True, exist_ok=True)

        # Extracting dataset information required for downloading from GEE
        (data, band, scale_factor, reducer, month_start_range,
         month_end_range, _, _) = self.get_gee_dict(data_name)

        # shift to user defined GEE reducer if 'ee_reducer' is not None.
        if ee_reducer is not None:
            logger.info(f'Using user-defined reducer {ee_reducer} ...')
            reducer = ee_reducer

        # loading input shape and extracting its total bounds
        download_bounds = gpd.read_file(input_shape_for_data_download).total_bounds

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for year in year_list:  # first loop for year_list
            for month in month_list:
                logger.info('---------------------------------------------------------------------------------')
                logger.info(f'Getting data urls for year={year}, month={month}.....')

                # Setting date ranges
                start_date = ee.Date.fromYMD(year, month, 1)
                start_date_dt = datetime(year, month, 1)

                if month < 12:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                    end_date_dt = datetime(year, month + 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                    end_date_dt = datetime(year + 1, 1, 1)

                # a condition to check whether start and end date falls in the available data range in GEE
                # if not the block will not be executed
                if (start_date_dt >= month_start_range) and (end_date_dt <= month_end_range):

                    # data filtering
                    if data_name == 'GRIDMET_RET':
                        # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                        # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                        # bias correction factor.

                        monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            reduce(reducer).multiply(0.85).multiply(scale_factor).toFloat().rename(band). \
                            reproject(crs='EPSG:4326', scale=scale_meters)

                    elif data_name == 'DAYMET_sunHr':
                        # dividing by 3600 to convert from second to hr

                        monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            reduce(reducer).divide(3600).multiply(scale_factor).toFloat().rename(band). \
                            reproject(crs='EPSG:4326', scale=scale_meters)

                    else:

                        monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            reduce(reducer).multiply(scale_factor).toFloat().rename(band). \
                            reproject(crs='EPSG:4326', scale=scale_meters)

                    # Download monthly image.
                    # The following block will check the 'number of pixels' within the requested bounds.
                    # Then, discretize the bound into manageable tile chunks and parallel process to download
                    # the entire data together

                    pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                                  scale_meters=scale_meters)


                    if pixels_in_bound > MAX_PIXELS_PER_TILE:
                        logger.info('Bounds too large for single download. Transitioning to tiled parallel processing and download.')

                        download_dir = Path(main_download_dir) / data_name / 'monthly'
                        download_dir.mkdir(parents=True, exist_ok=True)

                        self.__download_image_chunked(img=monthly_img,
                                                      bounds_coords=download_bounds,
                                                      scale_meters=scale_meters,
                                                      band_name=band,
                                                      data_name=data_name,
                                                      year=year, month=month,
                                                      download_dir=download_dir,
                                                      n_workers=use_cpu_while_multidownloading,
                                                      clip_resample_to_target_raster=clip_resample_to_target_raster,
                                                      clip_shapefile=input_shape_for_data_download,
                                                      ref_raster=ref_raster,
                                                      clip_resample_resolution=clip_resample_resolution)


                    else:
                        logger.info("Downloading as single tile...")

                        arr, coords = self.__download_single_tile(
                                                                 monthly_img,
                                                                 ee.Geometry.Rectangle(download_bounds.tolist()),
                                                                 band_name=band,
                                                                 default_nodata=0)

                        self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                           data_name, year, month)

                else:
                    logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                    pass
        
    def download_annual_LANID(self, data_name, input_shape_for_data_download, 
                              main_download_dir, year_list, scale_meters, 
                              use_cpu_while_multidownloading=15, 
                              skip_download=False):
        """
        Used to download LANID data from the provided URL.

        :param data_name: str, name of the dataset to download. Currently only supports 'LANID'.
        :param input_shape_for_data_download: str, file path of the input shape for data download.
        :param main_download_dir: str, file path of main download directory. It will consist directory of individual dataset.
        :param year_list: list of years to download data for.
        :param scale_meters: int, scale in meters for the downloaded data.
        :param skip_download: bool, set to True to skip download.

        :return: None
        """
        if not skip_download:
            
            # Initialize earth engine
            ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')   
            

            download_dir = Path(main_download_dir) / data_name / 'annual'

            download_dir.mkdir(parents=True, exist_ok=True)

            # loading input shape and extracting its total bounds
            download_bounds = gpd.read_file(input_shape_for_data_download).total_bounds


            for year in year_list:  # first loop for year_list
                logger.info('---------------------------------------------------------------------------------')
                logger.info(f'Getting data urls for year={year} .....')

                if 1997 <= year <= 2017:
                    # Extracting dataset information required for downloading from GEE
                    (data, band, _, _, _, _,
                    year_start_range, year_end_range) = self.get_gee_dict('LANID_1997_2017')
                
                    # Setting date ranges
                    start_date = ee.Date.fromYMD(year, 1, 1)
                    start_date_dt = datetime(year, 1, 1)

                    end_date = ee.Date.fromYMD(year, 12, 31)  
                    end_date_dt = datetime(year, 12, 31)

                    # a condition to check whether start and end date falls in the available data range in GEE
                    # if not the block will not be executed
                    if (start_date_dt >= year_start_range) and (end_date_dt <= year_end_range):
                        
                        # collecting the annual image
                        annual_lanid = ee.ImageCollection(data).filterDate(start_date, end_date). \
                            select(band).first().eq(1)
                        annual_lanid = annual_lanid.updateMask(annual_lanid) 
                        
                elif 2018 <= year <= 2025:
                    # Extracting dataset information required for downloading from GEE
                    (data, _, _, _, _, _,
                    year_start_range, year_end_range) = self.get_gee_dict('LANID_2018_2025')    

                    # Setting date ranges
                    start_date = ee.Date.fromYMD(year, 1, 1)
                    start_date_dt = datetime(year, 1, 1)

                    end_date = ee.Date.fromYMD(year, 12, 31)  
                    end_date_dt = datetime(year, 12, 31)
                    
                    # a condition to check whether start and end date falls in the available data range in GEE
                    # if not the block will not be executed
                    if (start_date_dt >= year_start_range) and (end_date_dt <= year_end_range):
                        
                        # collecting the annual image
                        band = 'irMap' + str(year)[-2:]
                        annual_lanid = ee.Image(data).select(band).eq(1)

                                    
                # Download annual image.
                # The following block will check the 'number of pixels' within the requested bounds.
                # Then, discretize the bound into manageable tile chunks and parallel process to download
                # the entire data together

                pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                                scale_meters=scale_meters)
                if pixels_in_bound > MAX_PIXELS_PER_TILE:
                    logger.info('Bounds too large for single download. Transitioning to tiled parallel processing and download.')

                    self.__download_image_chunked(img=annual_lanid,
                                                  bounds_coords=download_bounds,
                                                  scale_meters=scale_meters,
                                                  band_name=band,
                                                  data_name=data_name,
                                                  year=year, month=None,
                                                  download_dir=download_dir,
                                                  n_workers=use_cpu_while_multidownloading,
                                                  clip_resample_to_target_raster=False,
                                                  clip_shapefile=input_shape_for_data_download,
                                                  ref_raster=None,
                                                  clip_resample_resolution=None)     
        else:
            pass
        
        
    def download_gee_data_annual(self,
                                data_name: str,
                                main_download_dir: str,
                                input_shape_for_data_download: str,
                                year_list: list,
                                ee_reducer: ee.Reducer = None,
                                scale_meters: int = 2200,
                                clip_resample_to_target_raster: bool = True,
                                clip_resample_resolution: float = res_2km,
                                ref_raster: str = WestUS_raster,
                                use_cpu_while_multidownloading: int = 15
                                ) -> None:
        """
        Download annual data from Google Earth Engine (GEE) for the specified dataset and years.

        *** You can only download dataset declared in get_gee_dict()
        *** To download other datasets, please add the assets, band names, and start-end of data in get_gee_dict()
        *** Change 'ee_reducer' to your desired reducer if you don't prefer the default reducer from get_gee_dict()
        *** If the bounding region exceeds the Earth Engine pixel limit (MAX_PIXELS_PER_TILE), the function
            automatically switches to chunked parallel downloading.

        :param data_name : str
            Name of the dataset to download. Must be a valid key supported
            by `get_gee_dict()` (e.g., 'GRIDMET_Precip', 'GRIDMET_RET',
            'PRISM_Precip', etc.).

        :param main_download_dir : str or Path
            Directory where downloaded rasters will be saved.

        :param input_shape_for_data_download : str or Path
            Path to a shapefile defining the spatial extent to download.

        :param year_list : list[int]
            List of years to download.

        :param ee_reducer: ee.Reducer
            Use a desired reducer (e.g., ee.Reducer.mean()).
            Default set to None to use default reducer from get_gee_dict().

        :param scale_meters : int, optional
            Target spatial resolution in meters. Default is 2200.

        :param clip_resample_to_target_raster: bool
            If True, clip and resample output to reference raster. Set to False to skip this process and keep the
            original download intact.

        :param clip_resample_resolution float
            Resolution of the final data. Used in the process of clip & resample to make all downloaded data aligned
            with a reference raster.

        :param ref_raster : str or Path, optional
            Reference raster used for clipping and resampling.

        :param use_cpu_while_multidownloading : int, optional
            Number of parallel workers used during tiled download.
            Default is 15.

        :return None.
        """
        # Initialize earth engine
        ee.Initialize(project=self.ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

        download_dir = Path(main_download_dir) / data_name / 'annual'

        download_dir.mkdir(parents=True, exist_ok=True)

        # Extracting dataset information required for downloading from GEE
        (data, band, scale_factor, reducer, _, _, year_start_range,
            year_end_range) = self.get_gee_dict(data_name)

        # shift to user defined GEE reducer if 'ee_reducer' is not None.
        if ee_reducer is not None:
            logger.info(f'Using user-defined reducer {ee_reducer} ...')
            reducer = ee_reducer

        # loading input shape and extracting its total bounds
        download_bounds = gpd.read_file(input_shape_for_data_download).total_bounds


        for year in year_list:  # first loop for year_list
            logger.info('---------------------------------------------------------------------------------')
            logger.info(f'Getting data urls for year={year} .....')

            # Setting date ranges
            start_date = ee.Date.fromYMD(year, 1, 1)
            start_date_dt = datetime(year, 1, 1)

            end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
            end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= year_start_range) and (end_date_dt <= year_end_range):

                # data filtering
                if data_name == 'GRIDMET_RET':
                    # multiplying by 0.85 to applying bias correction in GRIDMET RET. GRIDMET RET is overestimated
                    # by 12-31% across CONUS (Blankenau et al. (2020). Senay et al. (2022) applied 0.85 as constant
                    # bias correction factor.

                    monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        reduce(reducer).multiply(0.85).multiply(scale_factor).toFloat().rename(band). \
                        reproject(crs='EPSG:4326', scale=scale_meters)

                elif data_name == 'DAYMET_sunHr':
                    # dividing by 3600 to convert from second to hr

                    monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        reduce(reducer).divide(3600).multiply(scale_factor).toFloat().rename(band). \
                        reproject(crs='EPSG:4326', scale=scale_meters)

                else:

                    monthly_img = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        reduce(reducer).multiply(scale_factor).toFloat().rename(band). \
                        reproject(crs='EPSG:4326', scale=scale_meters)

                # Download monthly image.
                # The following block will check the 'number of pixels' within the requested bounds.
                # Then, discretize the bound into manageable tile chunks and parallel process to download
                # the entire data together

                pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                                scale_meters=scale_meters)


                if pixels_in_bound > MAX_PIXELS_PER_TILE:
                    logger.info('Bounds too large for single download. Transitioning to tiled parallel processing and download.')

                    self.__download_image_chunked(img=monthly_img,
                                                  bounds_coords=download_bounds,
                                                  scale_meters=scale_meters,
                                                  band_name=band,
                                                  data_name=data_name,
                                                  year=year, month=None,
                                                  download_dir=download_dir,
                                                  n_workers=use_cpu_while_multidownloading,
                                                  clip_resample_to_target_raster=clip_resample_to_target_raster,
                                                  clip_shapefile=input_shape_for_data_download,
                                                  ref_raster=ref_raster,
                                                  clip_resample_resolution=clip_resample_resolution)


                else:
                    logger.info("Downloading as single tile...")

                    arr, coords = self.__download_single_tile(
                                                                monthly_img,
                                                                ee.Geometry.Rectangle(download_bounds.tolist()),
                                                                band_name=band,
                                                                default_nodata=0)

                    self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                        data_name, year)

            else:
                logger.warning(f'Data for year {year} is out of range. Skipping query')
                pass

        


def download_all_gee_data(ee_project, data_list, download_dir, year_list, month_range,
                          skip_download=False, use_cpu_while_multidownloading=15):
    """
    Used to download all gee data together.

    :param ee_project: Google Earth Engine project name to use for downloading data.
    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['GRIDMET_Precip', 'GRIDMET_Tmax', 'GRIDMET_RET', 'GRIDMET_maxRH',
        'GRIDMET_minRH', 'GRIDMET_windVel', 'GRIDMET_shortRad', 'GRIDMET_vpd',
        'DAYMET_sunHr', 'PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmean']

    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of year_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param skip_download: Set to True to skip download.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:

            if data_name in ['GRIDMET_Precip',
                             'GRIDMET_RET', 'GRIDMET_Tmax',
                             'GRIDMET_maxRH', 'GRIDMET_minRH',
                             'GRIDMET_windVel', 'GRIDMET_shortRad',
                             'GRIDMET_vpd', 'DAYMET_sunHr']:

                gee_download = GEE_download(ee_project=ee_project)
                gee_download.download_gee_data_monthly(
                    data_name=data_name,
                    main_download_dir=download_dir,
                    input_shape_for_data_download=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS.shp',
                    year_list=year_list,
                    month_range=month_range,
                    ee_reducer=None,    # will use default reducer from get_gee_dict
                    scale_meters=2200,
                    clip_resample_to_target_raster=True,
                    clip_resample_resolution=res_2km,
                    ref_raster=WestUS_raster,
                    use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name in ['PRISM_Precip', 'PRISM_Tmax', 'PRISM_Tmean']:
                gee_download = GEE_download(ee_project=ee_project)
                gee_download.download_gee_data_monthly(
                    data_name=data_name,
                    main_download_dir=download_dir,
                    input_shape_for_data_download=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS.shp',
                    year_list=year_list,
                    month_range=month_range,
                    ee_reducer=None,    # will use default reducer from get_gee_dict
                    scale_meters=2200,
                    clip_resample_to_target_raster=True,
                    clip_resample_resolution=res_2km,
                    ref_raster=WestUS_raster,
                    use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Peff_usda_scs':
                download_USDA_SCS_Peff_pycropwat(years_list=year_list,
                                                 output_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs',
                                                 scale_meters=2200,
                                                 ee_project=ee_project)
    else:
        pass

    
def download_USDA_SCS_Peff_pycropwat(years_list, output_dir, scale_meters, ee_project='ee-fahim'):
    """
    Download and process USDA-SCS effective precipitation (Peff) using pyCropWat.

    This function calculates monthly effective precipitation using the USDA-SCS
    soil moisture depletion method through the ``EffectivePrecipitation`` class
    in pyCropWat. It uses PRISM 800m precipitation data and OpenET ensemble
    reference evapotranspiration (ETo) data.

    Two different OpenET ETa assets are used depending on the year:

        - 1985–September 1999: OpenET provisional asset (v2_0_pre2000)
        - October 1999 onward: OpenET main asset (v2_0)

    References -
        pycropwat doc - https://montimaj.github.io/pyCropWat/
        pycropwat git repo - https://github.com/montimaj/pyCropWat/tree/main

    Parameters
    ----------
    years_list : list of int
        List of years to process.

    output_dir : str or Path
        Base directory for saving outputs. The following structure is created:

            output_dir/
                └── monthly/
                    ├── raw_download/
                    └── (final clipped rasters)

    scale_meters : float
        Spatial resolution (in meters) to use when downloading data from GEE.
        
    ee_project : str
        Google Earth Engine project name to use for downloading data. Default is 'ee-fahim

    Returns
    -------
        None
    """

    monthly_raw_output_dir = Path(output_dir) / 'monthly/raw_download'
    monthly_final_output_dir = Path(output_dir) / 'monthly'
    monthly_raw_output_dir.mkdir(parents=True, exist_ok=True)

    # extending the given year list to include the preceding year for the first year in the list. This is required for calculating Peff for the first year in the list.
    if len(years_list) > 0:
        first_year = years_list[0]
        preceding_year = first_year - 1
        if preceding_year not in years_list:
            years_list = [preceding_year] + years_list

    # USDA-SCS Required Assets/parameters
    # U.S. datasets (high-resolution SSURGO and GridMET)
    method_params_2000_onward = {
        'awc_asset': "projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite",
        'awc_band': None,  # single band, no band name required

        # OpenET main asset
        # full coverage from 1999 (October) onward
        'eto_asset': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'eto_band': 'et_ensemble_mad',
        'eto_is_daily': False,
        'eto_scale_factor': 1,

        'rooting_depth': 1
    }

    method_params_2000_preceeding = {
        'awc_asset': "projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite",
        'awc_band': None,  # single band, no band name required

        # OpenET provisional asset
        # full coverage from 1985 to 1999 (September)
        'eto_asset': 'projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0_pre2000',
        'eto_band': 'et_ensemble_mad',
        'eto_is_daily': False,
        'eto_scale_factor': 1,

        'rooting_depth': 1
    }

    for year in years_list:
        for month in range(1, 12 + 1):
            if (year >= 2000) or (year == 1999 and month in [10, 11, 12]):

                # Calculating USDA SCS effective precipitation through 'pycrowat' based on
                # PRISM 800m resolution, later downloading

                ep = EffectivePrecipitation(
                    asset_id='projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY',
                    precip_band='ppt',
                    geometry_path=WestUS_shape,
                    start_year=year,
                    end_year=year,
                    precip_scale_factor=1,
                    scale=scale_meters,
                    gee_project=ee_project,
                    method='usda_scs',
                    method_params=method_params_2000_onward
                    )

                ep.process(
                    output_dir=str(monthly_raw_output_dir),
                    n_workers=5,
                    months=[month],
                    save_inputs=False,
                    input_dir=None
                    )
            else:

                # Calculating USDA SCS effective precipitation through 'pycrowat' based on
                # PRISM 800m resolution, later downloading

                ep = EffectivePrecipitation(
                    asset_id='projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY',
                    precip_band='ppt',
                    geometry_path=str(PROJECT_ROOT / 'Data_main/ref_shapes/WestUS.shp'),
                    start_year=year,
                    end_year=year,
                    precip_scale_factor=1,
                    scale=scale_meters,
                    gee_project=ee_project,
                    method='usda_scs',
                    method_params=method_params_2000_preceeding
                    )

                ep.process(
                    output_dir=str(monthly_raw_output_dir),
                    n_workers=5,
                    months=[month],
                    save_inputs=False,
                    input_dir=None
                    )

    # Deleting Peff fraction datasets
    Peff_fraction_dataset = list(Path(monthly_raw_output_dir).glob('*fraction*.tif'))

    for file in Peff_fraction_dataset:
        file.unlink()

    # Clipping the downloaded raster to align with the reference raster
    all_tifs = list(Path(monthly_raw_output_dir).glob('*.tif'))
    Peff_datasets = [f for f in all_tifs if 'fraction' not in f.name]

    for peff in Peff_datasets:
        clip_resample_reproject_raster(input_raster=peff,
                                       input_shape=WestUS_shape,
                                       output_raster_dir=monthly_final_output_dir,
                                       clip_and_resample=True,
                                       ref_raster=WestUS_raster,
                                       resolution=res_2km)

    logger.info(f"Completed downloading and clipping rasters")