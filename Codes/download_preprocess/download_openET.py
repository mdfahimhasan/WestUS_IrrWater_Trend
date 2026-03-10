# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import ee
import sys
import time
import logging
import requests
import rasterio as rio
import geopandas as gpd
from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

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
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_states.shp'
WestUS_raster = PROJECT_ROOT / 'Data_main/ref_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = PROJECT_ROOT / 'Data_main/ref_rasters/GEE_merging_refraster_larger_grids.tif'


def get_openet_gee_dict(data_name, ee_project='ee-fahim'):
    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    gee_data_dict = {
        'OpenET_ensemble': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',  # 1999 (October onward)
        'OpenET_provisional': 'projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0_pre2000',  # full coverage from 1985 to 1999 (September)
        'USDA_CDL': 'USDA/NASS/CDL',
        'IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'LANID_1997_2020': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID',  # This LANID assessed was manually saved as custom GEE asset (from the original asset)
        'LANID_2021_2025': 'projects/routinelanid/assets/LANID/LANID2018-2025',   # New LANID asset for recent years
        'AIM-HPA': 'projects/h2yo/IrrigationMaps/AIM/AIM-HPA/AIM-HPA_Deines_etal_RSE_v01_extend_1984-2020',
        'Irrigation_Frac_IrrMapper': 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp',
        'Irrigation_Frac_LANID': 'projects/ee-fahim/assets/LANID_for_selected_states/selected_Annual_LANID'
    }

    gee_band_dict = {
        'OpenET_ensemble': 'et_ensemble_mad',
        'OpenET_provisional': 'et_ensemble_mad',
        'USDA_CDL': 'cropland',
        'IrrMapper': 'classification',
        'LANID_1997_2020': None,  # The data holds annual datasets in separate band. Will process it separately
        'LANID_2021_2025': None,
        'Irrigation_Frac_IrrMapper': 'classification',
        'AIM-HPA': None,
        'Irrigation_Frac_LANID': None  # The data holds annual datasets in separate band. Will process it out separately
    }

    gee_scale_dict = {
        'OpenET_ensemble': 1,
        'OpenET_provisional': 1,
        'USDA_CDL': 1,
        'IrrMapper': 1,
        'LANID_1997_2020': 1,
        'LANID_2021_2025': 1,
        'AIM-HPA': 1,
        'Irrigation_Frac_IrrMapper': 1,
        'Irrigation_Frac_LANID': 1
    }

    aggregation_dict = {
        'OpenET_ensemble': ee.Reducer.mean(),  # monthly data; doesn't matter whether use mean() or sum() as reducer. Change for yearly data download if needed.
        'OpenET_provisional': ee.Reducer.mean(),
        'USDA_CDL': ee.Reducer.first(),
        'IrrMapper': ee.Reducer.max(),
        'LANID_1997_2020': None,
        'LANID_2021_2025': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': ee.Reducer.max(),
        'Irrigation_Frac_LANID': None
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
        'LANID_1997_2020': None,
        'LANID_2021_2025': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(1986, 1, 1),
        'Irrigation_Frac_LANID': None
    }

    month_end_date_dict = {
        'OpenET_ensemble': datetime(2024, 12, 1),
        'OpenET_provisional': datetime(1999, 10, 1),  # 1984 only covers pacific-northwest
        'USDA_CDL': datetime(2023, 1, 1),
        'IrrMapper': datetime(2025, 1, 1),
        'LANID_1997_2020': None,
        'LANID_2021_2025': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(2025, 1, 1),
        'Irrigation_Frac_LANID': None
    }

    year_start_date_dict = {
        'OpenET_ensemble': datetime(1999, 10, 1),
        'OpenET_provisional': datetime(1985, 1, 1),  # 1984 only covers pacific-northwest
        'USDA_CDL': datetime(2008, 1, 1),  # CONUS/West US full coverage starts from 2008
        'IrrMapper': datetime(1986, 1, 1),
        'LANID_1997_2020': None,
        'LANID_2021_2025': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(1986, 1, 1),
        'Irrigation_Frac_LANID': None
    }

    year_end_date_dict = {
        'OpenET_ensemble': datetime(2024, 1, 1),
        'OpenET_provisional': datetime(1999, 10, 1),  # 1984 only covers pacific-northwest
        'USDA_CDL': datetime(2023, 1, 1),
        'IrrMapper': datetime(2025, 1, 1),
        'LANID_1997_2020': None,
        'LANID_2021_2025': None,
        'AIM-HPA': None,
        'Irrigation_Frac_IrrMapper': datetime(2025, 1, 1),
        'Irrigation_Frac_LANID': None
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
        month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
        year_end_date_dict[data_name]


def get_data_GEE_saveTopath(url_and_file_path):
    """
    Uses data url to get data from GEE and save it to provided local file paths.

    :param url_and_file_path: A list of tuples where each tuple has the data url (1st member) and local file path
                             (2nd member).
    :return: None
    """
    # unpacking tuple
    data_url, file_path = url_and_file_path

    # get data from GEE
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(data_url, allow_redirects=True)
            logger.info(f'Downloading {file_path} .....')

            r.raise_for_status()  # Raise an exception for bad HTTP status codes

            # save data to local file path
            with open(file_path, 'wb') as f:
                f.write(r.content)

            # This is a check block to see if downloaded datasets are OK
            # sometimes a particular grid's data is corrupted, but it's completely random, not sure why it happens.
            # Re-downloading the same data might not have that error

            file_path = Path(file_path)

            if file_path.suffix == '.tif':  # only for data downloaded in geotiff format
                src = rio.open(file_path)
                data = src.read(1)
                src.close()

            break  # exit loop if download and data reading succeeds; if data can't be read break will not be implemented
            # and code will go to the 'except' block

        except Exception as e:
            logger.warning(f'attempt {attempt + 1} failed for {file_path}. Error: {e}. Trying again')

            if attempt == MAX_RETRIES - 1:
                logger.warning(f'failed to download {file_path} after {MAX_RETRIES} attempts.')


def download_data_from_GEE_by_multiprocess(download_urls_fp_list, use_cpu=2):
    """
    Use python multiprocessing library to download data from GEE in a multi-thread approach. This function is a
    wrapper over get_data_GEE_saveTopath() function providing multi-threading support.

    :param download_urls_fp_list: A list of tuples where each tuple has the data url (1st member) and local file path
                                  (2nd member).
    :param use_cpu: Number of CPU/core (Int) to use for downloading. Default set to 2.

    :return: None.
    """
    # Using ThreadPool() instead of pool() as this is an I/O bound job not CPU bound
    # Using imap() as it completes assigning one task at a time to the ThreadPool()
    # and blocks until each task is complete
    logger.info('##########################################')
    logger.info('Downloading data from GEE..')
    logger.info(f'{cpu_count()} CPUs on this machine. Engaging {use_cpu} CPUs for downloading')
    logger.info('##########################################')

    pool = ThreadPool(use_cpu)
    results = pool.imap(get_data_GEE_saveTopath, download_urls_fp_list)
    pool.close()
    pool.join()


def download_openet_ensemble(download_dir, year_list, month_range, merge_keyword, grid_shape,
                             ee_project='ee-fahim', use_cpu_while_multidownloading=15, refraster_westUS=WestUS_raster,
                             refraster_gee_merge=GEE_merging_refraster_large_grids, westUS_shape=WestUS_shape):
    """
    Download openET ensemble data (at monthly scale) from GEE.

    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param ee_project: Earth Engine project name. Default is 'ee-fahim'.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param refraster_westUS: Reference raster to clip/save data for WestUS extent.
    :param refraster_gee_merge: Reference raster to use for merging downloaded datasets from GEE. The merged
                                datasets have to be clipped for Western US ROI.
    :param westUS_shape: Filepath of West US shapefile.

    :return: None.
    """
    global data_url

    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = Path(download_dir) / 'OpenET_ensemble'
    download_dir.mkdir(parents=True, exist_ok=True)

    # Extracting dataset information required for downloading from GEE
    openet_asset, band, multiply_scale, reducer, month_start_range, month_end_range, \
        year_start_range, year_end_range = get_openet_gee_dict('OpenET_ensemble')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='FID', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['FID']

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
                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    download_data = ee.ImageCollection(openet_asset).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we are adding the try-except block to
                    # retry failed connections
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data_url = download_data.getDownloadURL({'name': 'OpenET_ensemble',
                                                                     'crs': 'EPSG:4269',  # NAD83
                                                                     'scale': 2200,  # in meter. equal to ~0.02 deg
                                                                     'region': gee_extent,
                                                                     'format': 'GEO_TIFF'})
                            break  # if successful, exit the loop
                        except ee.EEException as e:
                            if attempt < max_retries - 1:
                                time.sleep(5)  # wait for 5 seconds before retrying
                                continue
                            else:
                                logger.warning(f"Failed to get data_url for year={year}, month={month}, grid={grid_sr}: {e}")
                                data_url = None

                    key_word = 'OpenET_ensemble'
                    local_file_path = Path(download_dir) / f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif'

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for j, k in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([j, k])

                        # Download data by multi-processing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

                mosaic_name = f'OpenET_ensemble_{year}_{month}.tif'
                mosaic_dir = Path(download_dir) / merge_keyword / 'merged'
                mosaic_dir.mkdir(parents=True, exist_ok=True)

                clip_dir = Path(download_dir) / merge_keyword
                clip_dir.mkdir(parents=True, exist_ok=True)

                search_by = f'*{year}_{month}*.tif'
                merged_arr, merged_raster = mosaic_rasters_from_directory(input_dir=download_dir,
                                                                          output_dir=mosaic_dir,
                                                                          raster_name=mosaic_name,
                                                                          ref_raster=refraster_gee_merge,
                                                                          search_by=search_by,
                                                                          nodata=no_data_value)

                clip_resample_reproject_raster(input_raster=merged_raster, input_shape=westUS_shape,
                                               output_raster_dir=clip_dir, clip_and_resample=True,
                                               use_ref_width_height=False, resolution=model_res,
                                               ref_raster=refraster_westUS)

                logger.info('OpenET_ensemble monthly data downloaded and merged')

            else:
                logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_Irr_frac_from_IrrMapper_yearly(data_name, download_dir, year_list, grid_shape,
                                            ee_project='ee-fahim', use_cpu_while_multidownloading=15):
    """
    Download IrrMapper Irrigated fraction data (at 2km scale) at yearly scale from GEE for 11 states in the Western US
    WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM for 1986-2024.

    ########################
    # READ ME (for Irrigation Data)

    ** This function only downloads the IrrMapper part of data.

    *** For downloading irrigated fraction data for ND, SD, OK, KS, NE, and TX use download_Irr_frac_from_LANID_yearly()
    function.

    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1986-2024,
    whereas LANID data consists datasets these 11 states and of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2025.
    AIM-HPA data covers the High Plains region from 1984 to 2020. For downloading LANID part of the data, we are combining
    AIM-HPA and LANID to capture unseen irrigated lands by either datasets, when both datasets are available.

    ########################
.
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrigation_Frac_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param ee_project: Earth Engine project name. Default is 'ee-fahim'.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global data_url

    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = Path(download_dir) / data_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, _, _, year_start_range, year_end_range = get_openet_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years_list
        logger.info('********************************')
        logger.info(f'Getting data urls for year={year} .....')

        start_date_dt = datetime(year, 1, 1)
        end_date_dt = datetime(year, 12, 31)

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) and (end_date_dt <= year_end_range):
            # Filtering data for the year range and reducing data
            irrmap_imcol = ee.ImageCollection(data)
            irrmap = irrmap_imcol.filter(ee.Filter.calendarRange(year, year, 'year')).select(band).reduce(reducer)

            # IrrMapper projection extraction
            projection_irrmap = ee.Image(irrmap_imcol.first()).projection()
            projection2km_scale = projection_irrmap.atScale(2200)

            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as null
            mask = irrmap.eq(0)
            irr_mask_only = irrmap.updateMask(mask).remap([0], [1]).setDefaultProjection(crs=projection_irrmap)

            # 30m Irrigation pixel count in each 2km pixel
            irr_pixel_count = irr_mask_only.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                .reproject(crs=projection2km_scale)
            # In IrrMapper dataset irrigated fields are assigned as 0
            # Converting the irrigated values to 1 and setting others as 0
            irr_mask_with_total = irrmap.eq(0).setDefaultProjection(crs=projection_irrmap)

            # Total number of 30m pixels count in each 2km pixel
            total_pixel_count = irr_mask_with_total.reduceResolution(reducer=ee.Reducer.count(), maxPixels=60000) \
                .reproject(crs=projection2km_scale)

            # counting fraction of irrigated lands in a pixel
            irrig_frac = irr_pixel_count.divide(total_pixel_count)

            # second loop for grids
            data_url_list = []
            local_file_paths_list = []

            for i in range(len(grid_no)):  # third loop for grids
                # converting grid geometry info to a GEE extent
                grid_sr = grid_no[i]
                roi = grid_geometry[i].bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Getting Data URl for each grid from GEE
                # The GEE connection gets disconnected sometimes, therefore, we are adding the try-except block to
                # retry failed connections
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                              'crs': 'EPSG:4269',  # NAD83
                                                              'scale': 2200,  # in meter. equal to ~0.02 deg
                                                              'region': gee_extent,
                                                              'format': 'GEO_TIFF'})
                        break  # if successful, exit the loop
                    except ee.EEException as e:
                        if attempt < max_retries - 1:
                            time.sleep(5)  # wait for 5 seconds before retrying
                            continue
                        else:
                            logger.warning(f"Failed to get data_url for year={year}, grid={grid_sr}: {e}")
                            data_url = None

                key_word = data_name
                local_file_path = Path(download_dir) / f'{key_word}_{str(year)}_{str(grid_sr)}.tif'

                # Appending data url and local file path (to save data) to a central list
                data_url_list.append(data_url)
                local_file_paths_list.append(local_file_path)

                # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                # there is enough data url gathered for download.
                if (len(data_url_list) == 120) | (
                        i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                    # Combining url and file paths together to pass in multiprocessing
                    urls_to_file_paths_compile = []
                    for j, k in zip(data_url_list, local_file_paths_list):
                        urls_to_file_paths_compile.append([j, k])

                    # Download data by multi-processing/multi-threading
                    download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                           use_cpu=use_cpu_while_multidownloading)

                    # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                    # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                    data_url_list = []
                    local_file_paths_list = []

        else:
            logger.warning(f'Data for year {year} is out of range. Skipping query')
            pass


def download_Irr_frac_from_LANID_yearly(data_name, download_dir, year_list, grid_shape,
                                        ee_project='ee-fahim', use_cpu_while_multidownloading=2):
    """
    Download LANID + AIM-HPA Irrigated fraction data (at 2km scale) at yearly scale from GEE for 6 states in
    the Western US ND, SD, OK, KS, NE, and TX for 1986 to 2025. For 2021-2025, only LANID data are available, while
    for 1986 to 1996, only AIM-HPA records are available.

    ########################
    # READ ME (for Irrigation Data)

    ** This function only downloads the LANID part of data.

    *** For downloading irrigated fraction data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM use
    download_Irr_frac_from_IrrMapper_yearly() function.

    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1986-2023,
    whereas LANID data consists datasets these 11 states and of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2025.
    AIM-HPA data covers the High Plains region from 1984 to 2020. For downloading LANID part of the data, we are
    combining AIM-HPA and LANID to capture unseen irrigated lands by either datasets (1997-2020), when both datasets
    are available.

    ########################
.
    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrigation_Frac_LANID']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param ee_project: Earth Engine project name. Default is 'ee-fahim'.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global irrig_frac, data_url

    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = Path(download_dir) / data_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    # Extracting irrigated (LANID + AIM-HPA) dataset information (saved as an asset) from GEE

    # LANID bands for 1997-2020
    lanid_asset_1997_2020, _, _, _, _, _, _, _ = get_openet_gee_dict('LANID_1997_2020')
    lanid_data_band_dict_1997_2020 = \
        {1997: 'lanid_1997', 1998: 'lanid_1998', 1999: 'lanid_1999', 2000: 'lanid_2000',
         2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003', 2004: 'lanid_2004',
         2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007', 2008: 'lanid_2008',
         2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011', 2012: 'lanid_2012',
         2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015', 2016: 'lanid_2016',
         2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019', 2020: 'lanid_2020'}

    # LANID bands for 2021-2025
    lanid_asset_2021_2025, _, _, _, _, _, _, _ = get_openet_gee_dict('LANID_2021_2025')
    lanid_data_band_dict_2021_2025 = \
        {2021: 'irMap21', 2022: 'irMap22', 2023: 'irMap23', 2024: 'irMap24', 2025: 'irMap25'}

    # AIM-HPA bands for 1986-2020
    aim_hpa_asset, _, _, _, _, _, _, _ = get_openet_gee_dict('AIM-HPA')
    aim_hpa_band_dict = {
        1986: 'b1986', 1987: 'b1987', 1988: 'b1988', 1989: 'b1989',
        1990: 'b1990', 1991: 'b1991', 1992: 'b1992', 1993: 'b1993',
        1994: 'b1994', 1995: 'b1995', 1996: 'b1996', 1997: 'b1997',
        1998: 'b1998', 1999: 'b1999', 2000: 'b2000', 2001: 'b2001',
        2002: 'b2002', 2003: 'b2003', 2004: 'b2004', 2005: 'b2005',
        2006: 'b2006', 2007: 'b2007', 2008: 'b2008', 2009: 'b2009',
        2010: 'b2010', 2011: 'b2011', 2012: 'b2012', 2013: 'b2013',
        2014: 'b2014', 2015: 'b2015', 2016: 'b2016', 2017: 'b2017',
        2018: 'b2018', 2019: 'b2019', 2020: 'b2020'
    }

    for year in year_list:  # first loop for years_list
        logger.info('********************************')
        logger.info(f'Getting data urls for year={year} .....')

        if year < 1997:  # downloading data from AIM-HPA for 1986 to 1996

            # AIM-HPA data for the year
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]

            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_aim_hpa_masked = irr_aim_hpa.selfMask()
            
            # 2km projection taken for AIM-HPA
            projection2km_scale = irr_aim_hpa.projection().atScale(2200)

            # 30m Irrigation pixel count in each 2km pixel
            irr_pixel_count = irr_aim_hpa_masked.reduceResolution(reducer=ee.Reducer.count(),
                                                           maxPixels=60000).reproject(crs=projection2km_scale)

            # Total number of 30m pixels in each 2km pixel
            total_pixel_count = irr_aim_hpa.unmask().reduceResolution(reducer=ee.Reducer.count(),
                                                           maxPixels=60000).reproject(crs=projection2km_scale)

            # counting fraction of irrigated lands in a pixel
            irrig_frac = irr_pixel_count.divide(total_pixel_count)

        elif 1997 <= year < 2021:  # downloading combined data from LANID and AIM-HPA for 2000-2020

            # # LANID data for the year
            # casting lanid band to aim-hpa band name for name harmonization that is needed in mosaicking
            # In LANID dataset irrigated fields are assigned as 1
            lanid_band = lanid_data_band_dict_1997_2020[year]
            irr_lanid = ee.Image(lanid_asset_1997_2020).select(lanid_band)
            irr_lanid = irr_lanid.eq(1)

            # 30m and 2km projection taken for LANID
            projection_lanid = irr_lanid.projection()
            projection2km_scale = irr_lanid.projection().atScale(2200)

            # AIM-HPA data for the year
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]
            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_aim_hpa = irr_aim_hpa.updateMask(irr_aim_hpa)
            irr_aim_hpa = irr_aim_hpa.rename([lanid_band])

            # Joining LANID and AIM-HPA
            # In irrigated (LANID + AIM-HPA) dataset irrigated fields are assigned as 1
            irr_total = ee.ImageCollection([irr_lanid, irr_aim_hpa]).mosaic()
            irr_total = irr_total.gt(0).setDefaultProjection(projection_lanid)

            # 30m Irrigation pixel count in each 2km pixel
            irr_pixel_count = irr_total.reduceResolution(reducer=ee.Reducer.count(),
                                                         maxPixels=60000).reproject(crs=projection2km_scale)

            # Unmasking() to keep the irrigated values to 1 and setting others as 0
            irr_total = irr_total.unmask()

            # Total number of 30m pixels in each 2km pixel
            total_pixel_count = irr_total.reduceResolution(reducer=ee.Reducer.count(),
                                                           maxPixels=60000).reproject(crs=projection2km_scale)

            # counting fraction of irrigated lands in a pixel
            irrig_frac = irr_pixel_count.divide(total_pixel_count)

        else:  # downloading data based on LANID for 2021-2024

            # # LANID data for the year
            irr_lanid = ee.Image(lanid_asset_2021_2025).select(lanid_data_band_dict_2021_2025[year]).eq(1)

            # 2km projection taken for LANID
            projection2km_scale = irr_lanid.projection().atScale(2200)

            # 30m Irrigation pixel count in each 2km pixel
            irr_pixel_count = irr_lanid.reduceResolution(reducer=ee.Reducer.count(),
                                                         maxPixels=60000).reproject(crs=projection2km_scale)

            # Unmasking() to keep the irrigated values to 1 and setting others as 0
            irr_total = irr_lanid.unmask()

            # Total number of 30m pixels in each 2km pixel
            total_pixel_count = irr_total.reduceResolution(reducer=ee.Reducer.count(),
                                                           maxPixels=60000).reproject(crs=projection2km_scale)

            # counting fraction of irrigated lands in a pixel
            irrig_frac = irr_pixel_count.divide(total_pixel_count)

        # second loop for grids
        data_url_list = []
        local_file_paths_list = []

        for i in range(len(grid_no)):  # third loop for grids
            # converting grid geometry info to a GEE extent
            grid_sr = grid_no[i]
            roi = grid_geometry[i].bounds
            gee_extent = ee.Geometry.Rectangle(roi)

            # Getting Data URl for each grid from GEE
            # The GEE connection gets disconnected sometimes, therefore, we
            # are adding the try-except block to retry failed connections
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data_url = irrig_frac.getDownloadURL({'name': data_name,
                                                          'crs': 'EPSG:4269',  # NAD83
                                                          'scale': 2200,  # in meter. equal to ~0.02 deg
                                                          'region': gee_extent,
                                                          'format': 'GEO_TIFF'})
                    break  # if successful, exit the loop
                except ee.EEException as e:
                    if attempt < max_retries - 1:
                        time.sleep(5)  # wait for 5 seconds before retrying
                        continue
                    else:
                        logger.warning(f"Failed to get data_url for year={year}, grid={grid_sr}: {e}")
                        data_url = None

            key_word = data_name
            local_file_path = Path(download_dir) / f'{key_word}_{str(year)}_{str(grid_sr)}.tif'

            # Appending data url and local file path (to save data) to a central list
            data_url_list.append(data_url)
            local_file_paths_list.append(local_file_path)

            # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
            # there is enough data url gathered for download.
            if (len(data_url_list) == 120) | (
                    i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                # Combining url and file paths together to pass in multiprocessing
                urls_to_file_paths_compile = []
                for j, k in zip(data_url_list, local_file_paths_list):
                    urls_to_file_paths_compile.append([j, k])

                # Download data by multi-processing/multi-threading
                download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                       use_cpu=use_cpu_while_multidownloading)

                # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                data_url_list = []
                local_file_paths_list = []


def download_Irr_CropET_from_OpenET_IrrMapper_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                      scale=2200, ee_project='ee-fahim', use_cpu_while_multidownloading=15):
    """
    Download irrigated cropET data (at monthly scale) from OpenET GEE by filtering ET data with irrigated field data from
    IrrMapper.

    ########################
    # READ ME (for Irrigated cropland Data)

    ** This function only downloads the IrrMapper part of data.

    *** For downloading irrigated cropET data for ND, SD, OK, KS, NE, and TX use
    download_Irr_CropET_from_OpenET_LANID_monthly() function.

    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1986-2024,
    whereas LANID data consists datasets these 11 states and of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2025.
    AIM-HPA data covers the High Plains region from 1984 to 2020. For downloading LANID part of the data, we are
    combining AIM-HPA and LANID to capture unseen irrigated lands by either datasets (1997-2020), when both datasets
    are available.

    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrig_crop_OpenET_IrrMapper']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param ee_project: Earth Engine project name. Default is 'ee-fahim'.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global openet_asset, data_url

    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = Path(download_dir) / data_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Extracting IrrMapper dataset information required for downloading from GEE
    irr_data, irr_band, irr_multiply_scale, irr_reducer, _, _, _, _ = get_openet_gee_dict('IrrMapper')

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    for year in year_list:  # first loop for years_list
        # # IrrMapper data for the year
        # In IrrMapper dataset irrigated fields are assigned as 0
        # Converting the irrigated values to 1 and setting others as nan
        # The mask will be applied on OpenET data to obtain cropET
        irrmap = ee.ImageCollection(irr_data).filter(ee.Filter.calendarRange(year, year, 'year')). \
            select(irr_band).reduce(irr_reducer)
        projection2km_scale = irrmap.projection().atScale(2200)  # 2km projection taken for IrrMapper

        irrig_filter = irrmap.eq(0)
        irr_mask = irrmap.updateMask(irrig_filter).remap([0], [1])

        for month in month_list:  # second loop for months

            # Extracting OpenET dataset information required for downloading from GEE

            # selecting open vs provisional data asset in GEE
            # openET 1985-1999 data is provisional and 2000 to upfront data in open in GEE
            # selecting appropriate OpenET GEE asset based on year
            if (year >= 2000) or (year == 1999 and month in [10, 11, 12]):
                openet_asset, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
                    _, _ = get_openet_gee_dict('OpenET_ensemble')

            else:
                openet_asset, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
                    _, _ = get_openet_gee_dict('OpenET_provisional')

            logger.info('********************************')
            logger.info(f'Getting data urls for year={year}, month={month}.....')

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
            if (start_date_dt >= et_month_start_range) and (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(openet_asset)
                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with Irrmapper irrigated data. This will set non-irrigated pixels' ET value to zero
                cropET_from_OpenET = openET_img.multiply(irr_mask)

                # averaging crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore, we are adding the try-except
                    # block to retry failed connections
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data_url = cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                          'crs': 'EPSG:4269',  # NAD83
                                                                          'scale': scale,
                                                                          # in meter. equal to ~0.02 deg
                                                                          'region': gee_extent,
                                                                          'format': 'GEO_TIFF'})
                            break  # if successful, exit the loop
                        except ee.EEException as e:
                            if attempt < max_retries - 1:
                                time.sleep(5)  # wait for 5 seconds before retrying
                                continue
                            else:
                                logger.warning(f"Failed to get data_url for year={year}, month={month}, grid={grid_sr}: {e}")
                                data_url = None

                    local_file_path = Path(download_dir) / f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif'

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-processing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_Irr_CropET_from_OpenET_LANID_monthly(data_name, download_dir, year_list, month_range, grid_shape,
                                                  scale=2200, ee_project='ee-fahim', use_cpu_while_multidownloading=15):
    """
    Download irrigated cropET data (at monthly scale) from OpenET GEE by filtering ET data with irrigated field data
    from LANID and  AIM-HPA.

    ########################
    # READ ME (for Irrigated cropland Data)

    ** This function only downloads the LANID part of data.

    *** For downloading irrigated cropET data for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM use
    download_Irr_CropET_from_OpenET_IrrMapper_monthly() function.

    IrrMapper Data is available for WA, OR, CA, ID, NV, UT, AZ, MT, WY, CO, and NM (11 states) for 1986-2023,
    whereas LANID data consists datasets these 11 states and of ND, SD, OK, KS, NE, and TX (06 states) for 1997-2025.
    AIM-HPA data covers the High Plains region from 1984 to 2020. For downloading LANID part of the data, we are
    combining AIM-HPA and LANID to capture unseen irrigated lands by either datasets (1997-2020), when both datasets
    are available.

    ########################

    :param data_name: Data name which will be used to extract GEE path, band, reducer, valid date range info from
                     get_gee_dict() function. Current valid data name is - ['Irrig_crop_OpenET_LANID']
    :param download_dir: File path of download directory.
    :param year_list: List of years_list to download data for. Should be within 2016 to 2020.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaicked.
    :param scale: Resolution (in m) at which data will be downloaded from earth engine. Default set to 2200 m.
    :param ee_project: Earth Engine project name. Default is 'ee-fahim'.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.

    :return: None.
    """
    global openet_asset, data_url

    ee.Initialize(project=ee_project, opt_url='https://earthengine-highvolume.googleapis.com')

    download_dir = Path(download_dir) / data_name
    download_dir.mkdir(parents=True, exist_ok=True)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry'].tolist()
    grid_no = grids['grid_no'].tolist()

    # creating list of months
    month_list = [m for m in range(month_range[0], month_range[1] + 1)]

    # Extracting irrigated (LANID + AIM-HPA) dataset information (saved as an asset) from GEE

    # LANID bands for 1997-2020
    lanid_asset_1997_2020, _, _, _, _, _, _, _ = get_openet_gee_dict('LANID_1997_2020')
    lanid_data_band_dict_1997_2020 = \
        {1997: 'lanid_1997', 1998: 'lanid_1998', 1999: 'lanid_1999', 2000: 'lanid_2000',
         2001: 'lanid_2001', 2002: 'lanid_2002', 2003: 'lanid_2003', 2004: 'lanid_2004',
         2005: 'lanid_2005', 2006: 'lanid_2006', 2007: 'lanid_2007', 2008: 'lanid_2008',
         2009: 'lanid_2009', 2010: 'lanid_2010', 2011: 'lanid_2011', 2012: 'lanid_2012',
         2013: 'lanid_2013', 2014: 'lanid_2014', 2015: 'lanid_2015', 2016: 'lanid_2016',
         2017: 'lanid_2017', 2018: 'lanid_2018', 2019: 'lanid_2019', 2020: 'lanid_2020'}

    # LANID bands for 2021-2025
    lanid_asset_2021_2025, _, _, _, _, _, _, _ = get_openet_gee_dict('LANID_2021_2025')
    lanid_data_band_dict_2021_2025 = \
        {2021: 'irMap21', 2022: 'irMap22', 2023: 'irMap23', 2024: 'irMap24', 2025: 'irMap25'}

    # AIM-HPA bands for 1986-2020
    aim_hpa_asset, _, _, _, _, _, _, _ = get_openet_gee_dict('AIM-HPA')
    aim_hpa_band_dict = {
        1986: 'b1986', 1987: 'b1987', 1988: 'b1988', 1989: 'b1989',
        1990: 'b1990', 1991: 'b1991', 1992: 'b1992', 1993: 'b1993',
        1994: 'b1994', 1995: 'b1995', 1996: 'b1996', 1997: 'b1997',
        1998: 'b1998', 1999: 'b1999', 2000: 'b2000', 2001: 'b2001',
        2002: 'b2002', 2003: 'b2003', 2004: 'b2004', 2005: 'b2005',
        2006: 'b2006', 2007: 'b2007', 2008: 'b2008', 2009: 'b2009',
        2010: 'b2010', 2011: 'b2011', 2012: 'b2012', 2013: 'b2013',
        2014: 'b2014', 2015: 'b2015', 2016: 'b2016', 2017: 'b2017',
        2018: 'b2018', 2019: 'b2019', 2020: 'b2020'
    }

    for year in year_list:  # first loop for years_list

        if year < 1997:  # downloading data from AIM-HPA for 1986 to 1996

            # AIM-HPA data for the year
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]

            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_total = irr_aim_hpa.updateMask(irr_aim_hpa)

            projection2km_scale = irr_aim_hpa.projection().atScale(2200)


        elif (year >= 1997) and (year < 2021):  # downloading combined data from LANID and AIM-HPA for 1997-2020

            # # LANID data for the year
            # In LANID dataset irrigated fields are assigned as 1
            lanid_band = lanid_data_band_dict_1997_2020[year]
            irr_lanid = ee.Image(lanid_asset_1997_2020).select(lanid_band)
            irr_lanid = irr_lanid.eq(1)

            # 30m and 2km projection taken for LANID
            projection_lanid = irr_lanid.projection()
            projection2km_scale = irr_lanid.projection().atScale(2200)

            # AIM-HPA data for the year
            aim_hpa = ee.Image(aim_hpa_asset)
            aim_hpa_band = aim_hpa_band_dict[year]
            irr_aim_hpa = aim_hpa.select(aim_hpa_band).eq(1)
            irr_aim_hpa = irr_aim_hpa.updateMask(irr_aim_hpa)
            irr_aim_hpa = irr_aim_hpa.rename([lanid_band])

            # Joining LANID and AIM-HPA
            # In irrigated (LANID + AIM-HPA) dataset irrigated fields are assigned as 1
            irr_total = ee.ImageCollection([irr_lanid, irr_aim_hpa]).mosaic()
            irr_total = irr_total.gt(0).setDefaultProjection(projection_lanid)

        else:  # downloading data from LANID for 2021-2024

            # # LANID data for the year
            irr_lanid = ee.Image(lanid_asset_2021_2025).select(lanid_data_band_dict_2021_2025[year]).eq(1)

            irr_total = irr_lanid.updateMask(irr_lanid)

            # 30m and 2km projection taken for LANID
            projection2km_scale = irr_lanid.projection().atScale(2200)

        # second loop for months
        for month in month_list:

            # Extracting OpenET dataset information required for downloading from GEE

            # selecting open vs provisional data asset in GEE
            # openET 1985-1999 data is provisional and 2000 to upfront data in open in GEE
            # selecting appropriate OpenET GEE asset based on year
            if (year >= 2000) or (year == 1999 and month in [10, 11, 12]):
                openet_asset, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
                    _, _ = get_openet_gee_dict('OpenET_ensemble')

            else:
                openet_asset, et_band, et_multiply_scale, et_reducer, et_month_start_range, et_month_end_range, \
                    _, _ = get_openet_gee_dict('OpenET_provisional')

            logger.info('********************************')
            logger.info(f'Getting data urls for year={year}, month={month}.....')

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
            if (start_date_dt >= et_month_start_range) and (end_date_dt <= et_month_end_range):
                openET_imcol = ee.ImageCollection(openet_asset)

                # getting default projection of OpenET
                projection_openET = ee.Image(openET_imcol.first()).projection()

                # getting image for year-month range.
                # the projection is lost during this image conversion, reapplying that at the end
                openET_img = openET_imcol.select(et_band).filterDate(start_date, end_date). \
                    reduce(et_reducer).multiply(et_multiply_scale).toFloat(). \
                    setDefaultProjection(crs=projection_openET)

                # multiplying OpenET with LANID irrigated data.
                cropET_from_OpenET = openET_img.multiply(irr_total)

                # averaging crop ET (from openET) from 30m to 2km scale
                cropET_from_OpenET = cropET_from_OpenET. \
                    reduceResolution(reducer=ee.Reducer.mean(), maxPixels=60000). \
                    reproject(crs=projection2km_scale)

                # will collect url and file name in url list and local_file_paths_list
                data_url_list = []
                local_file_paths_list = []

                for i in range(len(grid_no)):  # third loop for grids
                    # converting grid geometry info to a GEE extent
                    grid_sr = grid_no[i]
                    roi = grid_geometry[i].bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Getting Data URl for each grid from GEE
                    # The GEE connection gets disconnected sometimes, therefore,
                    # we are adding the try-except block to retry failed connections
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            data_url = \
                                cropET_from_OpenET.getDownloadURL({'name': data_name,
                                                                   'crs': 'EPSG:4269',  # NAD83
                                                                   'scale': scale,  # in meter. equal to ~0.02 deg
                                                                   'region': gee_extent,
                                                                   'format': 'GEO_TIFF'})
                            break  # if successful, exit the loop
                        except ee.EEException as e:
                            if attempt < max_retries - 1:
                                time.sleep(5)  # wait for 5 seconds before retrying
                                continue
                            else:
                                logger.warning(f"Failed to get data_url for year={year}, month={month}, grid={grid_sr}: {e}")
                                data_url = None

                    local_file_path = Path(download_dir) / f'{data_name}_{str(year)}_{str(month)}_{str(grid_sr)}.tif'

                    # Appending data url and local file path (to save data) to a central list
                    data_url_list.append(data_url)
                    local_file_paths_list.append(local_file_path)

                    # The GEE connection gets disconnected sometimes, therefore, we download the data in batches when
                    # there is enough data url gathered for download.
                    if (len(data_url_list) == 120) | (
                            i == len(grid_no) - 1):  # downloads data when one of the conditions are met
                        # Combining url and file paths together to pass in multiprocessing
                        urls_to_file_paths_compile = []
                        for i, j in zip(data_url_list, local_file_paths_list):
                            urls_to_file_paths_compile.append([i, j])

                        # Download data by multi-processing/multi-threading
                        download_data_from_GEE_by_multiprocess(download_urls_fp_list=urls_to_file_paths_compile,
                                                               use_cpu=use_cpu_while_multidownloading)

                        # After downloading some data in a batch, we empty the data_utl_list and local_file_paths_list.
                        # The empty lists will gather some new urls and file paths, and download a new batch of datasets
                        data_url_list = []
                        local_file_paths_list = []

            else:
                logger.warning(f'Data for year {year}, month {month} is out of range. Skipping query')
                pass


def download_openET_datasets(ee_project, data_list, download_dir, year_list, month_range,
                             grid_shape_for_2km_ensemble, grid_shape_for30m_irrmapper, grid_shape_for30m_lanid,
                             GEE_merging_refraster=GEE_merging_refraster_large_grids,
                             westUS_refraster=WestUS_raster, westUS_shape=WestUS_shape,
                             use_cpu_while_multidownloading=15, skip_download=False):
    """
    Used to download openET datasets from GEE.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['Irrig_crop_OpenET_IrrMapper', 'Irrig_crop_OpenET_LANID',
        'Irrigation_Frac_IrrMapper', 'Irrigation_Frac_LANID',
        'OpenET_ensemble']
        ******************************

    :param ee_project: Earth Engine project name.
    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years_list to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape_for_2km_ensemble: File path of larger grids to download data for Western US.
    :param grid_shape_for30m_irrmapper: File path of smaller grids to download data for IrrMapper extent and cropET from
                                        openET (these datasets are processed at 30m res in GEE, so smaller grids are
                                        required).
    :param grid_shape_for30m_lanid: File path of smaller grids to download data for LANID (for 6 central states) extent
                                    and cropET from openET (these datasets are processed at 30m res in GEE, so smaller
                                    grids are required).
    :param GEE_merging_refraster: Reference raster to mosaic openET ensemble 2km dataset.
    :param westUS_refraster: Western US reference raster.
    :param westUS_shape: Western US shapefile.
    :param use_cpu_while_multidownloading: Number (Int) of CPU cores to use for multi-download by
                                           multi-processing/multi-threading. Default set to 15.
    :param skip_download: Set to True to skip download.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:
            if data_name == 'OpenET_ensemble':
                download_openet_ensemble(download_dir=download_dir, year_list=year_list,
                                         month_range=month_range, merge_keyword='WestUS_monthly',
                                         grid_shape=grid_shape_for_2km_ensemble,
                                         use_cpu_while_multidownloading=15, refraster_westUS=westUS_refraster,
                                         refraster_gee_merge=GEE_merging_refraster,
                                         westUS_shape=westUS_shape, ee_project=ee_project)

            elif data_name == 'Irrig_crop_OpenET_IrrMapper':
                download_Irr_CropET_from_OpenET_IrrMapper_monthly(data_name=data_name, download_dir=download_dir,
                                                                  year_list=year_list, month_range=month_range,
                                                                  grid_shape=grid_shape_for30m_irrmapper, scale=2200,
                                                                  ee_project=ee_project, use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrig_crop_OpenET_LANID':
                download_Irr_CropET_from_OpenET_LANID_monthly(data_name=data_name, download_dir=download_dir,
                                                              year_list=year_list, month_range=month_range,
                                                              grid_shape=grid_shape_for30m_lanid, scale=2200,
                                                              ee_project=ee_project, use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrigation_Frac_IrrMapper':
                download_Irr_frac_from_IrrMapper_yearly(data_name=data_name, download_dir=download_dir,
                                                        year_list=year_list, grid_shape=grid_shape_for30m_irrmapper,
                                                        ee_project=ee_project, use_cpu_while_multidownloading=use_cpu_while_multidownloading)

            elif data_name == 'Irrigation_Frac_LANID':
                download_Irr_frac_from_LANID_yearly(data_name=data_name, download_dir=download_dir,
                                                    year_list=year_list, grid_shape=grid_shape_for30m_lanid,
                                                    use_cpu_while_multidownloading=use_cpu_while_multidownloading)
    else:
        pass

