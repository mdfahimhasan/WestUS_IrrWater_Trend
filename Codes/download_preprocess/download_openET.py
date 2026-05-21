# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu


import ee
import sys
import time
import logging
import requests
import numpy as np
import rasterio as rio
import geopandas as gpd
from pathlib import Path
from rasterio.transform import from_bounds
from datetime import datetime
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Optional

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utils.raster_ops import clip_resample_reproject_raster, mosaic_rasters_from_directory

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

# ***************************** earth engine authentication *****************************

# ee.Authenticate()

# ***************************************************************************************

no_data_value = -9999
res_2km = 0.01976293625031605786  # in deg, ~2 km
WestUS_raster      = PROJECT_ROOT / 'Data_main/ref_rasters/Western_US_refraster_2km.tif'
WestUS_shape       = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_states.shp'
IrrMapper_bounds_shape = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_gee_grid_for30m_IrrMapper.shp'
AIMHPA_bounds_shape    = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_gee_grid_for30m_LANID.shp'

# Maximum pixels per tile for GEE sampleRectangle (conservative limit)
# This number is decided based on trial-error to have the best optimum download performance within GEE quota limits.
MAX_PIXELS_PER_TILE = 3600  # 60 x 60

class GEE_download_OPENET:
    def __init__(self, ee_project: str = 'ee-fahim'):
        self.ee_project = ee_project
        self.high_volume_opt_url = 'https://earthengine-highvolume.googleapis.com'
        

    def get_openet_gee_dict(self, data_name):
        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        gee_data_dict = {
            'OpenET_ensemble': "projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0",  # 1999 (October onward)
            'OpenET_provisional': 'projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0_pre2000',  # full coverage from 1985 to 1999 (September)
            'USDA_CDL': 'USDA/NASS/CDL',
            'IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
            'LANID_1997_2017': 'users/xyhuwmir4/LANID_postCls/LANID_v2',
            'LANID_2018_2025': 'projects/routinelanid/assets/LANID/LANID2018-2025',   # New LANID asset for recent years
            'AIM-HPA': 'projects/h2yo/IrrigationMaps/AIM/AIM-HPA/AIM-HPA_Deines_etal_RSE_v01_extend_1984-2020',
            'Irrigation_Frac_Western': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
            'Irrigation_Frac_Eastern': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID'
        }

        gee_band_dict = {
            'OpenET_ensemble': 'et_ensemble_mad',
            'OpenET_provisional': 'et_ensemble_mad',
            'USDA_CDL': 'cropland',
            'IrrMapper': 'classification',
            'LANID_1997_2017': None, 
            'LANID_2018_2025': None,
            'Irrigation_Frac_Western': 'classification',
            'AIM-HPA': None,
            'Irrigation_Frac_Eastern': None  # The data holds annual datasets in separate band. Will process it out separately
        }

        gee_scale_dict = {
            'OpenET_ensemble': 1,
            'OpenET_provisional': 1,
            'USDA_CDL': 1,
            'IrrMapper': 1,
            'LANID_1997_2017': 1,
            'LANID_2018_2025': 1,
            'AIM-HPA': 1,
            'Irrigation_Frac_Western': 1,
            'Irrigation_Frac_Eastern': 1
        }

        aggregation_dict = {
            'OpenET_ensemble': ee.Reducer.mean(),  # monthly data; doesn't matter whether use mean() or sum() as reducer. Change for yearly data download if needed.
            'OpenET_provisional': ee.Reducer.mean(),
            'USDA_CDL': ee.Reducer.first(),
            'IrrMapper': ee.Reducer.max(),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None,
            'AIM-HPA': None,
            'Irrigation_Frac_Western': ee.Reducer.max(),
            'Irrigation_Frac_Eastern': None
        }

        # # Note on start date and end date dictionaries
        # The start and end dates have been set based on what duration of data can be downloaded.
        # They may not exactly match with the data availability in GEE
        # In most cases the end date is shifted a month later to cover the end month's data

        month_start_date_dict = {
            'OpenET_ensemble': datetime(1999, 10, 1),
            'OpenET_provisional': datetime(1985, 1, 1),  # 1984 only covers pacific-northwest
            'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
            'IrrMapper': datetime(1986, 1, 1),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None,
            'AIM-HPA': None,
            'Irrigation_Frac_Western': datetime(1986, 1, 1),
            'Irrigation_Frac_Eastern': None
        }

        month_end_date_dict = {
            'OpenET_ensemble': datetime(2025, 1, 1),
            'OpenET_provisional': datetime(1999, 10, 1),  # 1984 only covers pacific-northwest
            'USDA_CDL': datetime(2023, 1, 1),
            'IrrMapper': datetime(2025, 1, 1),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None,
            'AIM-HPA': None,
            'Irrigation_Frac_Western': datetime(2025, 1, 1),
            'Irrigation_Frac_Eastern': None
        }

        year_start_date_dict = {
            'OpenET_ensemble': datetime(1999, 10, 1),
            'OpenET_provisional': datetime(1985, 1, 1),  # 1984 only covers pacific-northwest
            'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
            'IrrMapper': datetime(1986, 1, 1),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None,
            'AIM-HPA': None,
            'Irrigation_Frac_Western': datetime(1986, 1, 1),
            'Irrigation_Frac_Eastern': None
        }

        year_end_date_dict = {
            'OpenET_ensemble': datetime(2025, 1, 1),
            'OpenET_provisional': datetime(1999, 10, 1),  # 1984 only covers pacific-northwest
            'USDA_CDL': datetime(2023, 1, 1),
            'IrrMapper': datetime(2025, 1, 1),
            'LANID_1997_2017': None,
            'LANID_2018_2025': None,
            'AIM-HPA': None,
            'Irrigation_Frac_Western': datetime(2025, 1, 1),
            'Irrigation_Frac_Eastern': None
        }

        return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
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
        lon_meters_per_degree = 111320 * np.cos(np.radians(mid_lat))

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
                               default_nodata: float = np.nan,
                               scale_meters: int = 2200,
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
                    No data value in the data. Default set to np.nan.


        :return: tuple or None
                 Tuple of (array, coordinates) of the image or None if download fails.
        """
        try:
            url = img.getDownloadURL({
                'name': band_name,
                'crs': 'EPSG:4269',
                'scale': scale_meters,
                'region': tile,
                'format': 'GEO_TIFF'
            })
            
            r = requests.get(url, timeout=1200); r.raise_for_status()  # check if the request was successful
            
            with rio.io.MemoryFile(r.content) as mf:
                with mf.open() as src:
                    arr = src.read(1).astype(np.float32)
                    b = src.bounds


            arr = np.array(arr, dtype=np.float32)
            coords = [(b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top)]

            return arr, coords

        except Exception as e:
            logger.warning(f'Failed to download tile: {e}')
            # logger.exception("Tile download crashed")
            return None
    
    @staticmethod
    def __save_raster_from_arr_coords(download_bounds: np.ndarray | list,
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


        with rio.open(
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
                                 clip_resample_to_target_raster: bool = False,
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
                     (img, tile, band_name, default_nodata, scale_meters)
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
                                               clip_and_resample=False,
                                               ref_raster=ref_raster,
                                               resolution=clip_resample_resolution)

            logger.info(f"Successfully mosaicked - clipped - resampled {len(tile_arrays)} {data_name} tiles.")
            logger.info('---------------------------------------------------------------------------------\n')

        except Exception as e:
            logger.warning(f"{data_name} chunked download failed: {e}.")

            raise


    def GEE_download_OPENET(self, data_name, main_download_dir, year_list,
                                 month_range, scale_meters=2200,
                                 use_cpu_while_multidownloading=5,
                                 clip_resample_to_target_raster=False,
                                 clip_resample_resolution=res_2km,
                                 input_shape_for_data_download=WestUS_shape,
                                 ref_raster=WestUS_raster):
        """
        Download openET ensemble data (at monthly scale) from GEE.


        :param download_dir: File path of download directory.
        :param year_list: List of years_list to download data for.
        :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
        :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
        :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                            multi-processing/multi-threading. Default set to 5.
        :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
        :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                    datasets have to be clipped for Western US ROI.
        :param input_shape_for_data_download: File path of the input shapefile for data download bounds. Default is set to WestUS_shape. 

        :return: None.
        """
        global data_url

        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        download_dir = Path(main_download_dir) / 'OpenET_ensemble'
        download_dir.mkdir(parents=True, exist_ok=True)

        # Extracting dataset information required for downloading from GEE
        openet_asset, band, multiply_scale, reducer, month_start_range, month_end_range, \
            year_start_range, year_end_range = self.get_openet_gee_dict('OpenET_ensemble')

        # loading input shape and extracting its total bounds
        download_bounds = gpd.read_file(input_shape_for_data_download).total_bounds


        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for year in year_list:  # first loop for years_list
            for month in month_list:  # second loop for months
                logger.info('********************************')
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
                    
                    monthly_img= ee.ImageCollection(openet_asset).select(band).filterDate(start_date, end_date) \
                                    .reduce(reducer).multiply(multiply_scale).toFloat()
                    
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


    def download_Irr_frac_for_western_region(self, data_name, main_download_dir, year_list,
                                             scale_meters=2200,
                                             use_cpu_while_multidownloading=5,
                                             pre_lanid_bounds_shape=IrrMapper_bounds_shape,
                                             lanid_bounds_shape=WestUS_shape):
        """
        Download Irrigated fraction (2km, yearly) from GEE for the 11 western states
        (WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM) for 1986 to 2025.
        For 1986-1996, IrrMapper is used over the western-only extent;
        for 1997-2025, LANID is used over the full WestUS extent
        (which makes this method the single source of LANID coverage for all 17 states).

        The 30m -> 2km reduceResolution happens server-side in GEE; the chunked download
        samples the resulting 2km image via the class's __download_image_chunked helper.

        :param data_name: Output sub-directory name (e.g., 'Irrigation_Frac_Western').
        :param main_download_dir: Root directory for downloads.
        :param year_list: List of years to download.
        :param scale_meters: Target download resolution in meters. Default 2200.
        :param use_cpu_while_multidownloading: Workers for tile-parallel download.
        :param clip_resample_resolution: Resolution (in degrees) for the clip-resample step.
        :param pre_lanid_bounds_shape: Bounds shapefile for 1986-1996 (IrrMapper extent).
        :param lanid_bounds_shape: Bounds shapefile for 1997-2025 (full WestUS extent).

        :return: None.
        """
        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        # download bounds
        download_cache = {
            'pre_lanid': gpd.read_file(pre_lanid_bounds_shape).total_bounds,
            'lanid': gpd.read_file(lanid_bounds_shape).total_bounds
        }

        # LANID bands for 1997-2017
        lanid_asset_1997_2017, _, _, _, _, _, _, _ = self.get_openet_gee_dict('LANID_1997_2017')
        lanid_data_band_dict_1997_2017 = \
            {1997: 'irMap97', 1998: 'irMap98', 1999: 'irMap99', 2000: 'irMap00',
            2001: 'irMap01', 2002: 'irMap02', 2003: 'irMap03', 2004: 'irMap04',
            2005: 'irMap05', 2006: 'irMap06', 2007: 'irMap07', 2008: 'irMap08',
            2009: 'irMap09', 2010: 'irMap10', 2011: 'irMap11', 2012: 'irMap12',
            2013: 'irMap13', 2014: 'irMap14', 2015: 'irMap15', 2016: 'irMap16',
            2017: 'irMap17'}

        # LANID bands for 2018-2025
        lanid_asset_2018_2025, _, _, _, _, _, _, _ = self.get_openet_gee_dict('LANID_2018_2025')
        lanid_data_band_dict_2018_2025 = \
            {2018: 'irMap18', 2019: 'irMap19', 2020: 'irMap20', 2021: 'irMap21', 
             2022: 'irMap22', 2023: 'irMap23', 2024: 'irMap24', 2025: 'irMap25'}

        for year in year_list:
            logger.info('********************************')
            logger.info(f'Building irrigated-fraction image for year={year} .....')

            irrig_frac = None  # ensure defined before downstream use

            # ------ Use IrrMapper for 1986-1996 ----------------------------------------
            if year < 1997:
                data, band, _, reducer, _, _, year_start_range, year_end_range = \
                    self.get_openet_gee_dict('IrrMapper')

                start_dt = datetime(year, 1, 1)
                end_dt = datetime(year, 12, 31)
                if not (start_dt >= year_start_range and end_dt <= year_end_range):
                    logger.warning(f'Year {year} is out of IrrMapper range. Skipping.')
                    continue

                irrmap_imcol = ee.ImageCollection(data)
                irrmap = irrmap_imcol.filter(ee.Filter.calendarRange(year, year, 'year')) \
                    .select(band).reduce(reducer)

                projection_irrmap = ee.Image(irrmap_imcol.first()).projection()
                projection2km_scale = projection_irrmap.atScale(scale_meters)

                # In IrrMapper irrigated pixels are 0 -> remap to 1, mask everything else
                mask = irrmap.eq(0)
                irr_mask_only = irrmap.updateMask(mask).remap([0], [1]) \
                    .setDefaultProjection(crs=projection_irrmap)

                irr_pixel_count = irr_mask_only.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irr_mask_with_total = irrmap.eq(0).setDefaultProjection(crs=projection_irrmap)
                total_pixel_count = irr_mask_with_total.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irrig_frac = irr_pixel_count.divide(total_pixel_count) \
                    .reproject(crs=projection2km_scale).rename('irrig_frac')
                    
                download_bounds = download_cache['pre_lanid']

            # ------ Use LANID for 1997-2017 -------------------------------------------
            elif 1997 <= year <= 2017:
                lanid_band = lanid_data_band_dict_1997_2017[year]
                irr_lanid = ee.Image(lanid_asset_1997_2017).select(lanid_band).eq(1)

                projection2km_scale = irr_lanid.projection().atScale(scale_meters)

                irr_pixel_count = irr_lanid.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irr_total = irr_lanid.unmask()
                total_pixel_count = irr_total.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irrig_frac = irr_pixel_count.divide(total_pixel_count) \
                    .reproject(crs=projection2km_scale).rename('irrig_frac')
                    
                download_bounds = download_cache['lanid']

            # ------ Use LANID for 2018-2025 -------------------------------------------
            else:
                irr_lanid = ee.Image(lanid_asset_2018_2025) \
                    .select(lanid_data_band_dict_2018_2025[year]).eq(1)

                projection2km_scale = irr_lanid.projection().atScale(scale_meters)

                irr_pixel_count = irr_lanid.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irr_total = irr_lanid.unmask()
                total_pixel_count = irr_total.reduceResolution(
                    reducer=ee.Reducer.count(), maxPixels=60000
                ).reproject(crs=projection2km_scale)

                irrig_frac = irr_pixel_count.divide(total_pixel_count) \
                    .reproject(crs=projection2km_scale).rename('irrig_frac')
                    
                download_bounds = download_cache['lanid']

            if irrig_frac is None:
                continue

            # Download via chunked downloader (helper handles tile mosaic + clip-resample)
            download_dir = Path(main_download_dir) / data_name / 'yearly'
            download_dir.mkdir(parents=True, exist_ok=True)

            # download
            pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                scale_meters=scale_meters)
            
            if pixels_in_bound > MAX_PIXELS_PER_TILE: 
                
                self.__download_image_chunked(img=irrig_frac,
                                            bounds_coords=download_bounds,
                                            scale_meters=scale_meters,
                                            band_name='irrig_frac',
                                            data_name=data_name,
                                            year=year, month=None,
                                            download_dir=download_dir,
                                            n_workers=use_cpu_while_multidownloading)
            
            else:
                logger.info("Downloading as single tile...")

                arr, coords = self.__download_single_tile(
                                                          irrig_frac,
                                                          ee.Geometry.Rectangle(download_bounds.tolist()),
                                                          band_name='irrig_frac',
                                                          default_nodata=np.nan)

                self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                    data_name, year, month=None)



    def download_Irr_frac_for_eastern_region(self, data_name, main_download_dir, year_list,
                                             scale_meters=2200,
                                             use_cpu_while_multidownloading=5,
                                             pre_lanid_bounds_shape=AIMHPA_bounds_shape):
        """
        Download Irrigated fraction (2km, yearly) from GEE for the High Plains region
        (eastern 6 states) using AIM-HPA for 1986-1996.

        For 1997-2025 the LANID-based irrigated fraction covers the entire WestUS extent and
        is downloaded by download_Irr_frac_for_western_region. This method silently skips
        years >= 1997.

        Server-side 30m -> 2km reduceResolution; chunked download via the class helper.

        :param data_name: Output sub-directory name (e.g., 'Irrigation_Frac_Eastern').
        :param main_download_dir: Root directory for downloads.
        :param year_list: List of years to download (only 1986-1996 years actually run).
        :param scale_meters: Target download resolution in meters. Default 2200.
        :param use_cpu_while_multidownloading: Workers for tile-parallel download.
        :param clip_resample_to_target_raster: If True, clip & resample to ref_raster. Default False
                                               (clip-resample is expected to happen in post-processing).
        :param clip_resample_resolution: Resolution (in degrees) for the clip-resample step.
        :param pre_lanid_bounds_shape: Bounds shapefile for 1986-1996 (AIM-HPA / HP extent).
        :param lanid_bounds_shape: Bounds shapefile for 1997-2025 (unused here; kept for API
                                   symmetry with the western method).
        :param ref_raster: Reference raster used by the clip-resample step.

        :return: None.
        """
        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        # AIM-HPA bands for 1986-2020 (only 1986-1996 is consumed here)
        aim_hpa_asset, _, _, _, _, _, _, _ = self.get_openet_gee_dict('AIM-HPA')
        aim_hpa_band_dict = {
            1986: 'b1986', 1987: 'b1987', 1988: 'b1988', 1989: 'b1989',
            1990: 'b1990', 1991: 'b1991', 1992: 'b1992', 1993: 'b1993',
            1994: 'b1994', 1995: 'b1995', 1996: 'b1996'
        }

        for year in year_list:
            # LANID (1997-2025) is downloaded once over the full WestUS extent by
            # download_Irr_frac_for_western_region. The eastern method only handles
            # the pre-LANID period (AIM-HPA, 1986-1996).
            if year >= 1997:
                logger.info(f'Year {year}: LANID covers the full WestUS extent; handled by '
                            f'download_Irr_frac_for_western_region. Skipping in the eastern method.')
                continue

            logger.info('********************************')
            logger.info(f'Building irrigated-fraction image for year={year} (AIM-HPA) .....')

            # ------ AIM-HPA for 1986-1996 (pre-LANID) ----------------------------------
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]

            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_aim_hpa_masked = irr_aim_hpa.selfMask()

            projection2km_scale = irr_aim_hpa.projection().atScale(scale_meters)

            irr_pixel_count = irr_aim_hpa_masked.reduceResolution(
                reducer=ee.Reducer.count(), maxPixels=60000
            ).reproject(crs=projection2km_scale)

            total_pixel_count = irr_aim_hpa.unmask().reduceResolution(
                reducer=ee.Reducer.count(), maxPixels=60000
            ).reproject(crs=projection2km_scale)

            irrig_frac = irr_pixel_count.divide(total_pixel_count) \
                .reproject(crs=projection2km_scale).rename('irrig_frac')

            # Download via chunked downloader (raw output; clip-resample happens in post-processing)
            download_dir = Path(main_download_dir) / data_name / 'yearly'
            download_dir.mkdir(parents=True, exist_ok=True)

            # Pre-LANID bounds for AIM-HPA (1986-1996 is the only branch reached)
            download_bounds = gpd.read_file(pre_lanid_bounds_shape).total_bounds

            pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                scale_meters=scale_meters)
            
            if pixels_in_bound > MAX_PIXELS_PER_TILE: 
                self.__download_image_chunked(img=irrig_frac,
                                            bounds_coords=download_bounds,
                                            scale_meters=scale_meters,
                                            band_name='irrig_frac',
                                            data_name=data_name,
                                            year=year, month=None,
                                            download_dir=download_dir,
                                            n_workers=use_cpu_while_multidownloading)
                
            else:
                
                logger.info("Downloading as single tile...")

                arr, coords = self.__download_single_tile(
                                                          irrig_frac,
                                                          ee.Geometry.Rectangle(download_bounds.tolist()),
                                                          band_name='irrig_frac',
                                                          default_nodata=np.nan)

                self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                    data_name, year, month=None)


    def download_Irr_CropET_from_OpenET_for_western_monthly(self, data_name, main_download_dir,
                                                            year_list, month_range,
                                                            scale_meters=2200,
                                                            use_cpu_while_multidownloading=5,
                                                            pre_lanid_bounds_shape=IrrMapper_bounds_shape,
                                                            lanid_bounds_shape=WestUS_shape):
        """
        Download irrigated cropET (2km, monthly) from OpenET GEE for the 11 western states
        by multiplying OpenET ET by an irrigated mask (IrrMapper for 1986-1996, LANID for
        1997-2025). Server-side 30m -> 2km reduceResolution; chunked download.

        Bounds switch per year: pre_lanid_bounds_shape for 1986-1996 (IrrMapper footprint),
        lanid_bounds_shape for 1997-2025 (full WestUS extent, since LANID covers everything).

        :param data_name: Output sub-directory name (e.g., 'Irrig_crop_OpenET_Western').
        :param main_download_dir: Root directory for downloads.
        :param year_list: List of years to download.
        :param month_range: Tuple (start_month, end_month).
        :param scale_meters: Target download resolution in meters. Default 2200.
        :param use_cpu_while_multidownloading: Workers for tile-parallel download.
        :param pre_lanid_bounds_shape: Bounds shapefile for 1986-1996 (IrrMapper extent).
        :param lanid_bounds_shape: Bounds shapefile for 1997-2025 (full WestUS extent).

        :return: None.
        """
        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        # IrrMapper info
        irr_data, irr_band, _, irr_reducer, _, _, _, _ = self.get_openet_gee_dict('IrrMapper')

          # LANID bands for 1997-2017
        lanid_asset_1997_2017, _, _, _, _, _, _, _ = self.get_openet_gee_dict('LANID_1997_2017')
        lanid_data_band_dict_1997_2017 = \
            {1997: 'irMap97', 1998: 'irMap98', 1999: 'irMap99', 2000: 'irMap00',
            2001: 'irMap01', 2002: 'irMap02', 2003: 'irMap03', 2004: 'irMap04',
            2005: 'irMap05', 2006: 'irMap06', 2007: 'irMap07', 2008: 'irMap08',
            2009: 'irMap09', 2010: 'irMap10', 2011: 'irMap11', 2012: 'irMap12',
            2013: 'irMap13', 2014: 'irMap14', 2015: 'irMap15', 2016: 'irMap16',
            2017: 'irMap17'}

        # LANID bands for 2018-2025
        lanid_asset_2018_2025, _, _, _, _, _, _, _ = self.get_openet_gee_dict('LANID_2018_2025')
        lanid_data_band_dict_2018_2025 = \
            {2018: 'irMap18', 2019: 'irMap19', 2020: 'irMap20', 2021: 'irMap21', 
             2022: 'irMap22', 2023: 'irMap23', 2024: 'irMap24', 2025: 'irMap25'}
            
            
        # Cache the two bounds (read each shape only once)
        bounds_cache = {
            'pre_lanid': gpd.read_file(pre_lanid_bounds_shape).total_bounds,
            'lanid':     gpd.read_file(lanid_bounds_shape).total_bounds,
        }

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]

        for year in year_list:
            # Build irrigated mask + 2km projection + bounds for the year
            if year < 1997:
                irrmap = ee.ImageCollection(irr_data) \
                    .filter(ee.Filter.calendarRange(year, year, 'year')) \
                    .select(irr_band).reduce(irr_reducer)
                projection2km_scale = irrmap.projection().atScale(scale_meters)
                irrig_filter = irrmap.eq(0)
                irr_mask = irrmap.updateMask(irrig_filter).remap([0], [1])
                download_bounds = bounds_cache['pre_lanid']

            elif 1997 <= year <= 2017:
                lanid_band = lanid_data_band_dict_1997_2017[year]
                irr_lanid = ee.Image(lanid_asset_1997_2017).select(lanid_band).eq(1)
                irr_mask = irr_lanid.updateMask(irr_lanid)
                projection2km_scale = irr_lanid.projection().atScale(scale_meters)
                download_bounds = bounds_cache['lanid']

            else:
                irr_lanid = ee.Image(lanid_asset_2018_2025) \
                    .select(lanid_data_band_dict_2018_2025[year]).eq(1)
                irr_mask = irr_lanid.updateMask(irr_lanid)
                projection2km_scale = irr_lanid.projection().atScale(scale_meters)
                download_bounds = bounds_cache['lanid']

            for month in month_list:
                # Pick OpenET asset based on year/month
                if (year >= 2000) or (year == 1999 and month in [10, 11, 12]):
                    openet_asset, et_band, et_multiply_scale, et_reducer, \
                        et_month_start_range, et_month_end_range, _, _ = \
                        self.get_openet_gee_dict('OpenET_ensemble')
                else:
                    openet_asset, et_band, et_multiply_scale, et_reducer, \
                        et_month_start_range, et_month_end_range, _, _ = \
                        self.get_openet_gee_dict('OpenET_provisional')

                logger.info('********************************')
                logger.info(f'Building cropET image for year={year}, month={month} .....')

                start_date = ee.Date.fromYMD(year, month, 1)
                start_date_dt = datetime(year, month, 1)

                if month < 12:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                    end_date_dt = datetime(year, month + 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                    end_date_dt = datetime(year + 1, 1, 1)

                if not (start_date_dt >= et_month_start_range and end_date_dt <= et_month_end_range):
                    logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                    continue

                openET_imcol = ee.ImageCollection(openet_asset)
                projection_openET = ee.Image(openET_imcol.first()).projection()

                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date) \
                    .reduce(et_reducer).multiply(et_multiply_scale).toFloat() \
                    .setDefaultProjection(crs=projection_openET)

                # Multiply OpenET by irrigated mask, then reduce 30m -> 2km
                cropET_from_OpenET = openET_img.multiply(irr_mask)
                cropET_from_OpenET = cropET_from_OpenET \
                    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000) \
                    .reproject(crs=projection2km_scale) \
                    .rename('cropET')

                # Download via chunked downloader (raw output; clip-resample in post-processing)
                download_dir = Path(main_download_dir) / data_name / 'monthly'
                download_dir.mkdir(parents=True, exist_ok=True)

                pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                                scale_meters=scale_meters)
            
                if pixels_in_bound > MAX_PIXELS_PER_TILE: 
                    self.__download_image_chunked(img=cropET_from_OpenET,
                                                bounds_coords=download_bounds,
                                                scale_meters=scale_meters,
                                                band_name='cropET',
                                                data_name=data_name,
                                                year=year, month=month,
                                                download_dir=download_dir,
                                                n_workers=use_cpu_while_multidownloading)
                    
                else:
                    logger.info("Downloading as single tile...")

                    arr, coords = self.__download_single_tile(cropET_from_OpenET,
                                                             ee.Geometry.Rectangle(download_bounds.tolist()),
                                                             band_name='cropET',
                                                             default_nodata=np.nan)

                    self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                        data_name, year, month=month)
                


    def download_Irr_CropET_from_OpenET_for_eastern_monthly(self, data_name, main_download_dir,
                                                            year_list, month_range,
                                                            scale_meters=2200,
                                                            use_cpu_while_multidownloading=5,
                                                            pre_lanid_bounds_shape=AIMHPA_bounds_shape):
        """
        Download irrigated cropET (2km, monthly) from OpenET GEE for the High Plains region
        (eastern 6 states) using AIM-HPA × OpenET for 1986-1996. Server-side 30m -> 2km
        reduceResolution; chunked download.

        For 1997-2025 the LANID-based cropET covers the entire WestUS extent and is downloaded
        by download_Irr_CropET_from_OpenET_for_western_monthly. This method silently skips
        years >= 1997.

        :param data_name: Output sub-directory name (e.g., 'Irrig_crop_OpenET_Eastern').
        :param main_download_dir: Root directory for downloads.
        :param year_list: List of years to download (only 1986-1996 years actually run).
        :param month_range: Tuple (start_month, end_month).
        :param scale_meters: Target download resolution in meters. Default 2200.
        :param use_cpu_while_multidownloading: Workers for tile-parallel download.
        :param pre_lanid_bounds_shape: Bounds shapefile for 1986-1996 (AIM-HPA / HP extent).

        :return: None.
        """
        ee.Initialize(project=self.ee_project, opt_url=self.high_volume_opt_url)

        # AIM-HPA bands for 1986-1996 (the only branch consumed here)
        aim_hpa_asset, _, _, _, _, _, _, _ = self.get_openet_gee_dict('AIM-HPA')
        aim_hpa_band_dict = {
            1986: 'b1986', 1987: 'b1987', 1988: 'b1988', 1989: 'b1989',
            1990: 'b1990', 1991: 'b1991', 1992: 'b1992', 1993: 'b1993',
            1994: 'b1994', 1995: 'b1995', 1996: 'b1996'
        }

        # Bounds for the pre-LANID period (AIM-HPA / High Plains)
        download_bounds = gpd.read_file(pre_lanid_bounds_shape).total_bounds

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]

        for year in year_list:
            # LANID (1997-2025) is downloaded once over the full WestUS extent by
            # download_Irr_CropET_from_OpenET_for_western_monthly. Skip here.
            if year >= 1997:
                logger.info(f'Year {year}: LANID-based cropET is handled by '
                            f'download_Irr_CropET_from_OpenET_for_western_monthly. '
                            f'Skipping in the eastern method.')
                continue

            # Build AIM-HPA irrigated mask for the year
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]
            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_total = irr_aim_hpa.updateMask(irr_aim_hpa)
            projection2km_scale = irr_aim_hpa.projection().atScale(scale_meters)

            for month in month_list:
                # Pick OpenET asset based on year/month
                if (year >= 2000) or (year == 1999 and month in [10, 11, 12]):
                    openet_asset, et_band, et_multiply_scale, et_reducer, \
                        et_month_start_range, et_month_end_range, _, _ = \
                        self.get_openet_gee_dict('OpenET_ensemble')
                else:
                    openet_asset, et_band, et_multiply_scale, et_reducer, \
                        et_month_start_range, et_month_end_range, _, _ = \
                        self.get_openet_gee_dict('OpenET_provisional')

                logger.info('********************************')
                logger.info(f'Building cropET image for year={year}, month={month} .....')

                start_date = ee.Date.fromYMD(year, month, 1)
                start_date_dt = datetime(year, month, 1)

                if month < 12:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                    end_date_dt = datetime(year, month + 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                    end_date_dt = datetime(year + 1, 1, 1)

                if not (start_date_dt >= et_month_start_range and end_date_dt <= et_month_end_range):
                    logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                    continue

                openET_imcol = ee.ImageCollection(openet_asset)
                projection_openET = ee.Image(openET_imcol.first()).projection()

                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date) \
                    .reduce(et_reducer).multiply(et_multiply_scale).toFloat() \
                    .setDefaultProjection(crs=projection_openET)

                # Multiply OpenET by irrigated mask, then reduce 30m -> 2km
                cropET_from_OpenET = openET_img.multiply(irr_total)
                cropET_from_OpenET = cropET_from_OpenET \
                    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000) \
                    .reproject(crs=projection2km_scale) \
                    .rename('cropET')

                # Download via chunked downloader (raw output; clip-resample in post-processing)
                download_dir = Path(main_download_dir) / data_name / 'monthly'
                download_dir.mkdir(parents=True, exist_ok=True)

                pixels_in_bound = self.__estimate_pixel_count(bounds_coords=download_bounds,
                                scale_meters=scale_meters)
            
                if pixels_in_bound > MAX_PIXELS_PER_TILE: 
                    self.__download_image_chunked(img=cropET_from_OpenET,
                                                bounds_coords=download_bounds,
                                                scale_meters=scale_meters,
                                                band_name='cropET',
                                                data_name=data_name,
                                                year=year, month=month,
                                                download_dir=download_dir,
                                                n_workers=use_cpu_while_multidownloading)
                    
                else:
                    logger.info("Downloading as single tile...")

                    arr, coords = self.__download_single_tile(cropET_from_OpenET,
                                                            ee.Geometry.Rectangle(download_bounds.tolist()),
                                                            band_name='cropET',
                                                            default_nodata=np.nan)

                    self.__save_raster_from_arr_coords(download_bounds, arr, main_download_dir,
                                                        data_name, year, month=month)


def download_openET_datasets(ee_project, data_list, main_download_dir, year_list, month_range,
                             input_shape_for_data_download=WestUS_shape,
                             pre_lanid_bounds_shape_western=IrrMapper_bounds_shape,
                             pre_lanid_bounds_shape_eastern=AIMHPA_bounds_shape,
                             lanid_bounds_shape=WestUS_shape,
                             scale_meters=2200,
                             use_cpu_while_multidownloading=5,
                             skip_download=False):
    """
    Dispatch downloads of OpenET-derived datasets via the GEE_download_OPENET class.

    Option-B layout:
      - 'OpenET_ensemble' downloads over `input_shape_for_data_download`.
      - 'Irrigation_Frac_Western' / 'Irrig_crop_OpenET_Western' download
        IrrMapper over `pre_lanid_bounds_shape_western` for 1986-1996, and
        LANID over `lanid_bounds_shape` for 1997-2025.
      - 'Irrigation_Frac_Eastern' / 'Irrig_crop_OpenET_Eastern' download
        AIM-HPA over `pre_lanid_bounds_shape_eastern` for 1986-1996 only;
        years >= 1997 are silently skipped because LANID is already covered
        end-to-end by the western methods.

    The 30m server-side processing for IrrMapper / LANID / AIM-HPA is preserved exactly as
    in the original per-grid code; only the download mechanism is replaced with the class's
    chunked tile downloader. Clip-resample is off by default — raw mosaicked outputs are
    expected to be stitched / clipped in a post-processing step.

    :param ee_project: Earth Engine project name.
    :param data_list: List of dataset names to download. Valid values:
        ['OpenET_ensemble',
         'Irrigation_Frac_Western', 'Irrigation_Frac_Eastern',
         'Irrig_crop_OpenET_Western', 'Irrig_crop_OpenET_Eastern']
    :param main_download_dir: Root directory for all downloads.
    :param year_list: List of years to download.
    :param month_range: Tuple (start_month, end_month). Used by monthly datasets only.
    :param input_shape_for_data_download: Shapefile defining the download bounds for the
                                          'OpenET_ensemble' dataset (default WestUS_states.shp).
    :param pre_lanid_bounds_shape_western: Bounds shapefile for 1986-1996 IrrMapper downloads
                                           (default WestUS_gee_grid_for30m_IrrMapper.shp).
    :param pre_lanid_bounds_shape_eastern: Bounds shapefile for 1986-1996 AIM-HPA downloads
                                           (default WestUS_gee_grid_for30m_LANID.shp,
                                           i.e. the eastern 6-state footprint).
    :param lanid_bounds_shape: Bounds shapefile for 1997-2025 LANID downloads
                               (default WestUS_states.shp — full 17-state extent).
    :param scale_meters: Target download resolution in meters. Default 2200.
    :param use_cpu_while_multidownloading: Workers for tile-parallel download.
    :param skip_download: If True, skip everything.

    :return: None.
    """
    if skip_download:
        return

    downloader = GEE_download_OPENET(ee_project=ee_project)

    # kwargs that every method understands (no bounds-shape kwargs; those vary per method)
    shared_kwargs = dict(
        main_download_dir=main_download_dir,
        year_list=year_list,
        scale_meters=scale_meters,
        use_cpu_while_multidownloading=use_cpu_while_multidownloading,
    )

    for data_name in data_list:
        if data_name == 'OpenET_ensemble':
            # Ensemble still uses a single bounds shape (no pre/post-LANID split)
            downloader.GEE_download_OPENET(
                data_name=data_name, month_range=month_range,
                input_shape_for_data_download=input_shape_for_data_download,
                **shared_kwargs,
            )

        elif data_name == 'Irrigation_Frac_Western':
            downloader.download_Irr_frac_for_western_region(
                data_name=data_name,
                pre_lanid_bounds_shape=pre_lanid_bounds_shape_western,
                lanid_bounds_shape=lanid_bounds_shape,
                **shared_kwargs,
            )

        elif data_name == 'Irrigation_Frac_Eastern':
            downloader.download_Irr_frac_for_eastern_region(
                data_name=data_name,
                pre_lanid_bounds_shape=pre_lanid_bounds_shape_eastern,
                **shared_kwargs,
            )
        

        elif data_name == 'Irrig_crop_OpenET_Western':
            downloader.download_Irr_CropET_from_OpenET_for_western_monthly(
                data_name=data_name, month_range=month_range,
                pre_lanid_bounds_shape=pre_lanid_bounds_shape_western,
                lanid_bounds_shape=lanid_bounds_shape,
                **shared_kwargs,
            )

        elif data_name == 'Irrig_crop_OpenET_Eastern':
            downloader.download_Irr_CropET_from_OpenET_for_eastern_monthly(
                data_name=data_name, month_range=month_range,
                pre_lanid_bounds_shape=pre_lanid_bounds_shape_eastern,
                **shared_kwargs,
            )

        else:
            logger.warning(f"Unknown data_name: {data_name}. Skipping.")
