# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import ast
import os
import re
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
from rasterio.mask import mask
from rasterstats import zonal_stats
from rasterio.warp import reproject, Resampling

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, sum_rasters, write_array_to_raster, \
    clip_resample_reproject_raster, shapefile_to_raster, mosaic_raster_list, rasterize_shape_to_match

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS.shp'
WestUS_raster = PROJECT_ROOT / 'Data_main/ref_rasters/Western_US_refraster_2km.tif'


# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


def extract_month_from_GrowSeason_data(GS_data_dir, skip_processing=False):
    """
    Extract start and ending growing season months from growing season dataset (provided by Justin Huntington DRI;
    downloaded from GEE to google drive). The output datasets have 2 all_bands, containing start and end month info,
    respectively.

    :param GS_data_dir: Directory path of growing season dataset. The GEE-downloaded datasets are in the
                        'ee_exports' folder.
    :param skip_processing: Set True if you want to skip processing.

    :return: None.
    """

    def doy_to_month(year, doy):
        """
        Convert a day of year (DOY) to a month in a given year.

        :return: Month of the corresponding date.
        """
        if np.isnan(doy):  # Check if the DOY is NaN
            return np.nan

        # January 1st of the given year + timedelta of the DoY to extract month
        month = (datetime.datetime(year, 1, 1) + datetime.timedelta(int(doy) - 1)).month

        return month

    if skip_processing:
        return
    
    # collecting GEE exported data files and making new directories for processing
    GS_data_dir = Path(GS_data_dir)
    GS_data_files = list((GS_data_dir / 'ee_exports').glob('*.tif'))
    interim_dir = GS_data_dir / 'interim'
    interim_dir.mkdir(parents=True, exist_ok=True)

    # looping through each dataset, extracting start and end of the growing season months, saving as an array
    for data in GS_data_files:
        raster_name = data.name
        year = int(raster_name.split('_')[1].split('.')[0])

        logging.info(f'Processing growing season data for {year}...')

        # clipping and resampling the growing season data with the western US reference raster
        interim_raster = clip_resample_reproject_raster(input_raster=data,
                                                        input_shape=WestUS_shape,
                                                        raster_name=raster_name,
                                                        output_raster_dir=interim_dir,
                                                        clip=False, resample=False, clip_and_resample=True,
                                                        targetaligned=True, resample_algorithm='near',
                                                        use_ref_width_height=False, ref_raster=None,
                                                        resolution=model_res)

        # reading the start and end DoY of the growing season
        startDOY_arr, ras_file = read_raster_arr_object(interim_raster, band=1)
        endDOY_arr = read_raster_arr_object(interim_raster, band=2, get_file=False)

        # vectorizing the doy_to_month() function to apply on a numpy array
        vectorized_doy_to_date = np.vectorize(doy_to_month)

        # converting the start and end DoY to corresponding month
        start_months = vectorized_doy_to_date(year, startDOY_arr)
        end_months = vectorized_doy_to_date(year, endDOY_arr)

        # stacking the arrays together (single tif with 2 all_bands)
        GS_month_arr = np.stack((start_months, end_months), axis=0)

        # saving the array
        output_raster = GS_data_dir / raster_name

        with rio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=GS_month_arr.shape[1],
                width=GS_month_arr.shape[2],
                dtype=np.float32,
                count=GS_month_arr.shape[0],
                crs=ras_file.crs,
                transform=ras_file.transform,
                nodata=-9999
        ) as dst:
            dst.write(GS_month_arr)


def sum_vars_water_yr(years_list, var_monthly_dir, output_dir_water_yr,
                      save_keyword, skip_processing=False):
    """
    Sum any variable for for water year.


    :param years_list: Tuple/list of years_list for which data will be processed.
    :param var_monthly_dir: Directory file path of monthly datasets of the variable of interest.
    :param output_dir_water_yr: File path of directory to save summed variable for each water year.
    :param save_keyword: Keyword to use for summed cropET data saving.
    :param skip_processing: Set True to skip processing.

    :return: None.
    """
    if skip_processing:
        return
    
    output_dir_water_yr = Path(output_dir_water_yr)
    output_dir_water_yr.mkdir(parents=True, exist_ok=True)
    
    var_monthly_dir = Path(var_monthly_dir)

    for yr in years_list:
        logger.info(f'summing monthly {save_keyword} dataset for water year {yr}...')

        # summing rainfed/irrigated crop ET for water year (previous year's October to current year's september)
        et_data_prev_years = [f for f in var_monthly_dir.glob(f'*{yr - 1}*tif')
                            if any(f'_{m}.' in f.name or f'_0{m}.' in f.name for m in [10, 11, 12])]

        et_data_current_years = [f for f in var_monthly_dir.glob(f'*{yr}*tif')
                                if any(f'_{m}.' in f.name or f'_0{m}.' in f.name for m in range(1, 10))]

        if not et_data_prev_years:
            # for 1986 water year aggregation, this block will be executed because there is no data for 1985 
            # (previous year) for some datasets. 
            
            logger.warning(f'No data for {yr-1} — computing water year {yr} from current year only.')
            et_water_yr_list = et_data_current_years 
            
        else:
            et_water_yr_list = et_data_prev_years + et_data_current_years
            
        sum_rasters(raster_list=et_water_yr_list, raster_dir=None,
                    output_raster=output_dir_water_yr / f'{save_keyword}_{yr}.tif',
                    ref_raster=et_water_yr_list[0])

def sum_precip_winter_months(years_list, var_monthly_dir, output_dir,
                             save_keyword, skip_processing=False):
    """
    Sum winter months of precipitation.

    :param years_list: Tuple/list of years_list for which data will be processed.
    :param var_monthly_dir: Directory file path of monthly precipitation datasets of the variable of interest.
    :param output_dir: File path of directory to save summed variable for each water year.
    :param save_keyword: Keyword to use for data saving.
    :param skip_processing: Set True to skip processing.

    :return: None.
    """
    if skip_processing:
        return
        
    output_dir_water_yr = Path(output_dir)
    output_dir_water_yr.mkdir(parents=True, exist_ok=True)
    
    var_monthly_dir = Path(var_monthly_dir)

    for yr in years_list:
        logger.info(f'summing monthly {save_keyword} dataset for winter months (November-March) for {yr}...')

        # summing precipitation for winter months (previous year's November to current year's March)
        data_prev_years = [f for f in var_monthly_dir.glob(f'*{yr - 1}*tif')
                            if any(f'_{m}.' in f.name or f'_0{m}.' in f.name for m in [11, 12])]

        data_current_years = [f for f in var_monthly_dir.glob(f'*{yr}*tif')
                                if any(f'_{m}.' in f.name or f'_0{m}.' in f.name for m in range(1, 4))]

        if not data_prev_years:                
            logger.warning(f'No data for {yr-1} — computing water year {yr} from current year only.')
            winter_month_data_list = data_current_years 
            
        else:
            winter_month_data_list = data_prev_years + data_current_years
            
        sum_rasters(raster_list=winter_month_data_list, raster_dir=None,
                    output_raster=output_dir_water_yr / f'{save_keyword}_{yr}.tif',
                    ref_raster=winter_month_data_list[0])


def dynamic_gs_sum_of_variable(year_list, growing_season_dir, monthly_input_dir,
                               gs_output_dir, sum_keyword,
                               ref_raster=WestUS_raster,
                               skip_processing=False):
    """
    Dynamically (spatio-temporally) sums any variable for dynamic growing seasons.

    :param year_list: List of years_list to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for summed growing season datasets.
    :param sum_keyword: Keyword str to add before the summed raster.
    :param ref_raster: Filepath of reference raster. Default set to Western US 2km reference raster.
    :param skip_processing: Set True if you want to skip processing this step.

    :return: None.
    """
    if skip_processing:
        return

    monthly_input_dir = Path(monthly_input_dir)
    gs_output_dir = Path(gs_output_dir)
    gs_output_dir.mkdir(parents=True, exist_ok=True)

    # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
    # from the filenames using the first group ([0-9]{1,2}).
    # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
    # for proper sorting by month.
    month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

    for year in year_list:
        logger.info(f'Dynamically summing {sum_keyword} monthly datasets for growing season {year}...')

        # gathering and sorting the datasets by month (from 1 to 12)
        datasets = list(monthly_input_dir.glob(f'*{year}*.tif'))
        sorted_datasets = sorted(
            datasets,
            key=lambda x: int(month_pattern.search(x.name).group(1)))  # First capturing group (the month)

        # monthly array stacked in a single numpy array
        arrs_stck = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets], axis=0)

        # gathering, reading, and stacking growing season array
        gs_data = list(Path(growing_season_dir).glob(f'*{year}*.tif'))[0]
        start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)  # band 1
        end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)  # band 2

        # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
        # Then it is broadcasted across the array and named as the kernel_mask.
        # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
        kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
        kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

        # sum monthly arrays over the valid months using the kernel_mask
        summed_arr = np.nansum(arrs_stck * kernel_mask, axis=0)

        # in some cases nan positions are changed to zero because of np.nan sum
        # reinstating them as -9999 with reference raster
        ref_arr = read_raster_arr_object(ref_raster, get_file=False)
        summed_arr = np.where(np.isnan(ref_arr), -9999, summed_arr)

        # saving the summed array
        output_path = gs_output_dir / f'{sum_keyword}_{year}.tif'

        with rio.open(
                output_path,
                'w',
                driver='GTiff',
                height=summed_arr.shape[0],
                width=summed_arr.shape[1],
                dtype=np.float32,
                count=1,
                crs=ras_file.crs,
                transform=ras_file.transform,
                nodata=-9999
        ) as dst:
            dst.write(summed_arr, 1)

    logger.info('All dynamic summing completed')
    logger.info('---------------------------------------------------------------')


def dynamic_gs_mean_of_variable(year_list, growing_season_dir, monthly_input_dir, gs_output_dir,
                                mean_keyword, skip_processing=False):
    """
    Dynamically (spatio-temporally) averages any variable for dynamic growing seasons.

    :param year_list: List of years to process the data for.
    :param growing_season_dir: Directory path for growing season datasets.
    :param monthly_input_dir:  Directory path for monthly datasets.
    :param gs_output_dir:  Directory path (output) for averaged growing season datasets.
    :param mean_keyword: Keyword str to add before the averaged raster.
    :param skip_processing: Set True if  you want to skip processing this step.

    :return: None.
    """
    if skip_processing:
        return

    monthly_input_dir = Path(monthly_input_dir)
    gs_output_dir = Path(gs_output_dir)
    gs_output_dir.mkdir(parents=True, exist_ok=True)

    month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

    for year in year_list:
        logger.info(f'Dynamically averaging {mean_keyword} monthly datasets for growing season {year}...')

        # gathering and sorting the datasets by month (from 1 to 12)
        datasets = list(monthly_input_dir.glob(f'*{year}*.tif'))
        sorted_datasets = sorted(datasets, key=lambda x: int(month_pattern.search(x.name).group(1)))

        # monthly array stacked in a single numpy array
        arrs_stck = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_datasets], axis=0)

        # gathering, reading, and stacking growing season array
        gs_data = list(Path(growing_season_dir).glob(f'*{year}*.tif'))[0]
        start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)
        end_gs_arr = read_raster_arr_object(gs_data, band=2, get_file=False)

        # We create a 1 pixel "kernel", representing months 1 to 12 (shape : 12, 1, 1).
        # Then it is broadcasted across the array and named as the kernel_mask.
        # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
        kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
        kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

        # Count the number of valid months in each pixel's growing season
        valid_month_count = np.sum(kernel_mask, axis=0)
        valid_month_count = valid_month_count.astype(
            'float')  # converting valid_month_count to float to allow np.nan assignment
        valid_month_count[
            valid_month_count == 0] = np.nan  # to avoid division by zero for non-growing season pixels

        # computing the mean over valid months
        summed_arr = np.sum(arrs_stck * kernel_mask, axis=0)
        mean_arr = summed_arr / valid_month_count

        # saving the mean array
        output_path = gs_output_dir / f'{mean_keyword}_{year}.tif'

        with rio.open(
                output_path,
                'w',
                driver='GTiff',
                height=mean_arr.shape[0],
                width=mean_arr.shape[1],
                dtype=np.float32,
                count=1,
                crs=ras_file.crs,
                transform=ras_file.transform,
                nodata=-9999
        ) as dst:
            dst.write(mean_arr, 1)

    logger.info('All dynamic averaging completed')
    logger.info('---------------------------------------------------------------')


def paste_and_reproject(src_raster_path, ref_raster_path, nodata):
    """
    Reproject a source raster (small extent, possibly different CRS/resolution)
    into the reference raster grid.
    Returns a full-size array aligned to reference raster.
    """

    # opening the reference raster file and creating an empty array using its shape
    ref_profile = rio.open(ref_raster_path)
    out_arr = np.full((ref_profile.height, ref_profile.width), nodata, dtype=np.float32)

    # read the smaller array (src_raster_path)
    src_profile = rio.open(src_raster_path)
    src_arr = src_profile.read(1)

    # reproject the src array to the crs and pixel size of the reference raster
    reproject(
        source=src_arr,
        destination=out_arr,
        src_transform=src_profile.transform,
        src_crs=src_profile.crs,
        dst_transform=ref_profile.transform,
        dst_crs=ref_profile.crs,
        resampling=Resampling.nearest,
        dst_nodata=nodata
    )

    return out_arr, ref_profile


def merge_GEE_data_patches_IrrMapper_LANID_extents(year_with_full_extent, input_dir_irrmapper,
                                                   input_dir_lanid, merged_output_dir,
                                                   merge_keyword, year_with_partial_extent=None,
                                                   monthly_data=True, ref_raster=WestUS_raster,
                                                   skip_processing=False):
    """
    Merge/mosaic downloaded GEE data for IrrMapper and LANID extent.

    :param year_with_full_extent: Tuple/list of years for which data will be processed. This list should be
                                  used to process datasets for which data is available for entire Western US extent.

    :param input_dir_irrmapper: Input directory filepath of datasets at IrrMapper extent.
    :param input_dir_lanid: Input directory filepath of datasets at LANID extent.
    :param merged_output_dir: Output directory filepath to save merged data.
    :param merge_keyword: Keyword to use while merging. Foe example: 'Rainfed_Frac', 'Irrigated_crop_OpenET', etc.
    :param year_with_partial_extent: Tuple/list of years for which partial data is available for the Western US extent.
                                Default is None.
    :param monthly_data: Boolean. If False will look/search for yearly data patches. Default set to True to look for
                         monthly datasets.
    :param ref_raster: Reference raster to use in merging. Default set to Western US reference raster.
    :param skip_processing: Set True to skip merging IrrMapper and LANID extent data patches.

    :return: None.
    """
    if not skip_processing:
        input_dir_irrmapper = Path(input_dir_irrmapper)
        input_dir_lanid = Path(input_dir_lanid)
        merged_output_dir = Path(merged_output_dir)
        merged_output_dir.mkdir(parents=True, exist_ok=True)

        ################################################################################################################
        # # processing block for monthly data like ET
        if monthly_data:  # for datasets that are monthly
            month_list = list(range(1, 13))

            # processing block for data that has Western US-scale coverage
            if year_with_full_extent is not None:
                for year in year_with_full_extent:
                    for month in month_list:
                        search_by = f'*{year}_{month}_*.tif'

                        # making input raster list by joining rasters of irrmapper extent and rasters of lanid extent
                        irrmapper_raster_list = list(input_dir_irrmapper.glob(search_by))
                        lanid_raster_list = list(input_dir_lanid.glob(search_by))
                        irrmapper_raster_list.extend(lanid_raster_list)

                        total_raster_list = irrmapper_raster_list

                        if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                            merged_raster_name = f'{merge_keyword}_{year}_{month}.tif'
                            mosaic_raster_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                               raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                               resampling_method='nearest', mosaicking_method='first',
                                               resolution=model_res, nodata=no_data_value)

                            logger.info(f'{merge_keyword} data merged for year {year}, month {month}')

            # processing block for data that has partial coverage over Western US
            if year_with_partial_extent is not None:
                for year in year_with_partial_extent:
                    for month in month_list:
                        search_by = f'*{year}_{month}_*.tif'

                        ref_arr, ref_file = read_raster_arr_object(ref_raster)

                        # Opening each data and pasting it on ref raster.
                        # This approach is followed because we want to create a WesternUS-wide raster even if
                        # there is no irrigated cropland data for LANID and AIM-HPA after 2020 for midwest and
                        # CONUS-wide
                        irrmapper_raster_list = list(input_dir_irrmapper.glob(search_by))

                        for irr_data in irrmapper_raster_list:
                            temp_arr, _ = paste_and_reproject(src_raster_path=irr_data,
                                                              ref_raster_path=ref_raster,
                                                              nodata=no_data_value)

                            ref_arr = np.where(temp_arr != -9999, temp_arr, ref_arr)

                        ref_arr[ref_arr == 0] = no_data_value
                        output_raster = merged_output_dir / f'{merge_keyword}_{year}_{month}.tif'
                        write_array_to_raster(ref_arr, ref_file, ref_file.transform, output_raster)

                        logger.info(f'{merge_keyword} data merged for year {year}, month {month}')

        ################################################################################################################
        # # processing block for datasets that are yearly like land use
        else:

            # processing block for data that has Western US-scale coverage
            if year_with_full_extent is not None:
                for year in year_with_full_extent:
                    search_by = f'*{year}_*.tif'

                    # making input raster list by joining rasters of irrmapper extent and rasters of lanid extent
                    irrmapper_raster_list = list(input_dir_irrmapper.glob(search_by))
                    lanid_raster_list = list(input_dir_lanid.glob(search_by))
                    irrmapper_raster_list.extend(lanid_raster_list)

                    total_raster_list = irrmapper_raster_list

                    if len(total_raster_list) > 0:  # to only merge for years_list and months when data is available
                        merged_raster_name = f'{merge_keyword}_{year}.tif'
                        mosaic_raster_list(input_raster_list=total_raster_list, output_dir=merged_output_dir,
                                           raster_name=merged_raster_name, ref_raster=ref_raster, dtype=None,
                                           resampling_method='nearest', mosaicking_method='first',
                                           resolution=model_res, nodata=no_data_value)

                        logger.info(f'{merge_keyword} data merged for year {year}')

            # processing block for data that has partial coverage over Western US
            if year_with_partial_extent is not None:
                for year in year_with_partial_extent:
                    search_by = f'*{year}_*.tif'

                    ref_arr, ref_file = read_raster_arr_object(ref_raster)

                    # Opening each data and pasting it on ref raster.
                    # This approach is followed because we want to create a WesternUS-wide raster even if
                    # there is no irrigated cropland data for LANID and AIM-HPA after 2020 for midwest and CONUS-wide
                    irrmapper_raster_list = list(input_dir_irrmapper.glob(search_by))

                    for irr_data in irrmapper_raster_list:
                        temp_arr, _ = paste_and_reproject(src_raster_path=irr_data,
                                                          ref_raster_path=ref_raster,
                                                          nodata=no_data_value)

                        ref_arr = np.where(temp_arr != -9999, temp_arr, ref_arr)

                    ref_arr[ref_arr == 0] = no_data_value
                    output_raster = merged_output_dir / f'{merge_keyword}_{year}.tif'
                    write_array_to_raster(ref_arr, ref_file, ref_file.transform, output_raster)

                    logger.info(f'{merge_keyword} data merged for year {year}')
    else:
        pass


def classify_irrigated_cropland(years, irrigated_fraction_dir,
                                irrigated_cropland_output_dir,
                                basin_range_shp,
                                skip_processing=False):
    """
    Classifies irrigated cropland using irrigated fraction data.

    Three separate irrigated fraction thresholds are applied depending on the year and region,
    reflecting a systematic difference in detection sensitivity between the underlying
    datasets:

        - 1997 and later  : LANID + AIM-HPA combined fraction → threshold = 0.13 (13%)
        - Pre-1997        : AIM-HPA only fraction             → threshold = 0.08 (8%)
        - basin & range region: All years                        → threshold = 0.01 (1%)

    ########################
    # THRESHOLD DECISION NOTES

    ** Why 13% for >=1997
    The irrigated fraction for 1997-2020 is derived from a combination of LANID and
    AIM-HPA datasets (see download_Irr_frac_from_LANID_yearly). The 13% threshold
    was determined from prior calibration against reference irrigated cropland data
    for the LANID+AIM-HPA combined product.

    ** Why 8% for pre-1997
    ----------------------
    Pre-1997 irrigated fraction data relies solely on AIM-HPA (Deines et al.), which
    exhibits systematically lower irrigated fraction values compared to the LANID+AIM-HPA
    combined product. Visual cross-check analysis in GEE using overlap years (1997-2020),
    where both datasets are independently available, showed:

        1. The gap between AIM-HPA-only and LANID+AIM-HPA is largest in the early
           overlap years (1997-2001), which are temporally closest to the pre-1997
           period. At threshold=0.13, the combined product classified ~10-15% more
           pixels as irrigated than AIM-HPA alone in these years.
        2. Threshold calibration curves (% pixels irrigated vs threshold) consistently
           showed that an AIM-HPA threshold of ~0.08-0.10 produces irrigated area
           estimates equivalent to the combined product at 0.13 across multiple
           Kansas test years (1997, 1998, 1999, 2000, 2002, 2008, 2015).
        3. The scatter plot of AIM-HPA fraction vs combined fraction showed a trendline
           slope of ~0.9, meaning AIM-HPA runs ~10% lower on average — translating
           0.13 to approximately 0.11-0.12. The additional adjustment to 0.08 accounts
           for the larger gap observed specifically in the early LANID years (1997-2001)
           that are adjacent to the pre-1997 period.

    Using 0.08 for pre-1997 minimises the artificial discontinuity in classified
    irrigated area at the 1996/1997 boundary introduced by the dataset transition.

    ** Basin & Range exception:
    A lower threshold of 0.01 is applied uniformly across all years within the Basin
    and Range region (defined by basin_range_shp), regardless of the year-based
    threshold above. This region has distinct irrigation patterns that require a
    separate classification rule.

    ########################

    :param years: List of years to process data for.
    :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
    :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
    :param basin_range_shp: Basin and range-fill region shapefile. Pixels within this
                            region are classified at a lower threshold (>0.01) regardless
                            of year.
    :param skip_processing: Set True to skip classifying irrigated and rainfed cropland data.

    :return: None
    """
    if skip_processing:
        return
    
    irrigated_fraction_dir = Path(irrigated_fraction_dir)
    irrigated_cropland_output_dir = Path(irrigated_cropland_output_dir)
    irrigated_cropland_output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        logger.info(f'Classifying irrigated cropland data for year {year}')

        irrigated_frac_data = irrigated_fraction_dir / f'Irrigated_Frac_{year}.tif'
        irrig_arr, irrig_file = read_raster_arr_object(irrigated_frac_data)

        # create mask raster for Basin & Range shapefile extent
        basin_range_mask = rasterize_shape_to_match(input_shape=basin_range_shp,
                                                    ref_raster=irrigated_frac_data,
                                                    burn_value=1, fill_value=0)

        # empty array to store cropland classification
        irrigated_cropland = np.full_like(irrig_arr, -9999, dtype=np.int32)

        # Basin & Range pixels classified at lower threshold (0.01) across all years
        irrigated_cropland = np.where((basin_range_mask == 1) & (irrig_arr > 0.01), 1,
                                        irrigated_cropland)

        if year >= 1997:
            # LANID+AIM-HPA combined product — standard 13% threshold
            irrigated_cropland = np.where((basin_range_mask == 0) & (irrig_arr > 0.13), 1,
                                            irrigated_cropland)
        else:
            # AIM-HPA only — reduced 8% threshold to compensate for right-skewed
            # fraction distribution and lower detection sensitivity vs LANID+AIM-HPA.
            # See docstring for full decision rationale.
            irrigated_cropland = np.where((basin_range_mask == 0) & (irrig_arr > 0.08), 1,
                                            irrigated_cropland)

        # saving classified data
        output_irrigated_cropland_raster = irrigated_cropland_output_dir / f'Irrigated_cropland_{year}.tif'

        write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file,
                                transform=irrig_file.transform,
                                output_path=output_irrigated_cropland_raster,
                                dtype=np.int32)
        
    
def calculate_monthly_IWU(years_list, irrigated_cropET_monthly_dir, peff_monthly_dir,
                          iwu_output_dir, skip_processing=False):
    """
    Calculate monthly Irrigation Water Use (IWU) by subtracting effective precipitation (Peff; USDA-SCS method)
    from irrigated crop ET. Three versions are computed based on how many prior months of
    Peff are averaged:

        Version 1 — current month Peff only:
            IWU_{m} = ET_{m} - Peff_{m}

        Version 2 — average of current and previous month:
            IWU_{m} = ET_{m} - mean(Peff_{m}, Peff_{m-1})

        Version 3 — average of current and two prior months:
            IWU_{m} = ET_{m} - mean(Peff_{m}, Peff_{m-1}, Peff_{m-2})

    
    Note:
    -------
    - IWU is only computed where both ET and Peff are valid (not nodata) and ET > 0.
    - IWU is only estiamted for April (4) to October (10) as these are generally the growing season months in the Western US.
    - Multi-month Peff averaging (v2 and v3) accounts for the lag between when precipitation falls and when
        it reduces irrigation demand. If prior-month data is unavailable (e.g., Jan of the first
        year in years_list and no prior-year data exists on disk), the average is computed from
        whatever months are available.

    Output folder structure:
        iwu_output_dir/
        └── IWU_monthly/
            ├── peff_v1_current/           (Version 1)
            ├── peff_v2_current_prev1/     (Version 2)
            └── peff_v3_current_prev2/     (Version 3)

    :param years_list: List of years to process.
    :param irrigated_cropET_monthly_dir: Directory containing monthly irrigated crop ET rasters.
    :param peff_monthly_dir: Directory containing monthly Peff rasters.
    :param iwu_output_dir: Root output directory. Subfolders are created automatically.
    :param skip_processing: Set True to skip this step.

    :return: None.
    """
    if skip_processing:
        return
    
    # -------------------------------------------------------------------------
    # directory setup
    # -------------------------------------------------------------------------
    iwu_output_dir = Path(iwu_output_dir)
    irrigated_cropET_monthly_dir = Path(irrigated_cropET_monthly_dir)
    peff_monthly_dir = Path(peff_monthly_dir)

    out_v1 = iwu_output_dir / 'IWU_monthly' / 'peff_v1_current'
    out_v2 = iwu_output_dir / 'IWU_monthly' / 'peff_v2_current_prev1'
    out_v3 = iwu_output_dir / 'IWU_monthly' / 'peff_v3_current_prev2'

    for d in [out_v1, out_v2, out_v3]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # helper: step n months back from (year, month)
    # -------------------------------------------------------------------------
    def prev_month(year, month, n=1):
        """Return (year, month) that is n months before the given date."""
        m = month - n
        y = year
        while m <= 0:
            m += 12
            y -= 1
        return y, m

    # -------------------------------------------------------------------------
    # helper: load a Peff array; return None if file not found
    # -------------------------------------------------------------------------
    def load_peff(year, month):
        matches = list(peff_monthly_dir.glob(f'*{year}_{month:02d}.tif'))
        if not matches:
            logger.warning(f'Peff file not found for year={year}, month={month} — skipping contribution.')
            return None
        return read_raster_arr_object(matches[0], get_file=False)

    # -------------------------------------------------------------------------
    # helper: average a list of valid Peff arrays, respecting nodata
    # -------------------------------------------------------------------------
    def mean_peff(peff_arrays):
        """
        Average a list of Peff arrays. nodata (-9999) is excluded from the mean.
        Pixels that are nodata in ALL contributing arrays remain -9999.
        """
        valid_stack = []
        for arr in peff_arrays:
            if arr is None:
                continue
            masked = np.where(arr == no_data_value, np.nan, arr.astype(np.float32))
            valid_stack.append(masked)

        if not valid_stack:
            return None

        stacked = np.stack(valid_stack, axis=0)  # shape: (n_months, H, W)
        mean_arr = np.nanmean(stacked, axis=0)    # NaN where all inputs were nodata

        # restore -9999 where all inputs were nodata
        all_nan = np.all(np.isnan(stacked), axis=0)
        mean_arr = np.where(all_nan, no_data_value, mean_arr)

        return mean_arr.astype(np.float32)

    # -------------------------------------------------------------------------
    # helper: compute IWU = ET - peff_avg, with validity mask and neg-value check
    # -------------------------------------------------------------------------
    def compute_iwu(et_arr, peff_avg_arr, year, month, version_label):
        """
        Subtract peff_avg from ET where both are valid.
        Flags and zeros out negative IWU pixels.
        Returns IWU array.
        """
        et_f   = et_arr.astype(np.float32)
        peff_f = peff_avg_arr.astype(np.float32)

        # valid mask: both ET and Peff must be non-nodata
        valid = (et_f != no_data_value) & (et_f > 0) & (peff_f != no_data_value)

        iwu = np.full_like(et_f, no_data_value, dtype=np.float32)
        iwu[valid] = et_f[valid] - peff_f[valid]
        iwu[valid & (iwu < 0)] = 0  # set negative IWU to zero as it is not physically meaningful (precip suffices crop water demand)

        return iwu
    
    # -------------------------------------------------------------------------
    # main processing loop
    # -------------------------------------------------------------------------
    for year in years_list:
        
        # only compute for April (4) to October (10) - in general growing season in Western US
        for month in range(4, 11): 

            logger.info(f'Computing monthly IWU for year={year}, month={month:02d}...')

            # locate and load irrigated crop ET
            et_matches = list(irrigated_cropET_monthly_dir.glob(f'*{year}_{month}.tif'))
            if not et_matches:
                logger.warning(f'Irrigated crop ET file not found for year={year}, month={month} — skipping.')
                continue

            et_arr, ras_file = read_raster_arr_object(et_matches[0], get_file=True)

            # ------------------------------------------------------------------
            # load Peff for current and up to 2 prior months
            # ------------------------------------------------------------------
            peff_m0 = load_peff(year, month)

            y1, m1 = prev_month(year, month, n=1)
            peff_m1 = load_peff(y1, m1)

            y2, m2 = prev_month(year, month, n=2)
            peff_m2 = load_peff(y2, m2)

            # ------------------------------------------------------------------
            # Version 1: current month only
            # ------------------------------------------------------------------
            peff_v1 = mean_peff([peff_m0])
            if peff_v1 is not None:
                iwu_v1 = compute_iwu(et_arr, peff_v1, year, month, version_label='v1_current')
                out_path_v1 = out_v1 / f'IWU_{year}_{month}.tif'
                write_array_to_raster(iwu_v1, ras_file, ras_file.transform, out_path_v1, 
                                      nodata=no_data_value)

            else:
                logger.warning(f'Skipping v1 for year={year}, month={month} — no valid Peff.')

            # ------------------------------------------------------------------
            # Version 2: current + previous month
            # ------------------------------------------------------------------
            peff_v2 = mean_peff([peff_m0, peff_m1])
            if peff_v2 is not None:
                iwu_v2 = compute_iwu(et_arr, peff_v2, year, month, version_label='v2_current_prev1')
                out_path_v2 = out_v2 / f'IWU_{year}_{month}.tif'
                write_array_to_raster(iwu_v2, ras_file, ras_file.transform, out_path_v2, 
                                      nodata=no_data_value)
                
            else:
                logger.warning(f'Skipping v2 for year={year}, month={month} — no valid Peff.')

            # ------------------------------------------------------------------
            # Version 3: current + 2 prior months
            # ------------------------------------------------------------------
            peff_v3 = mean_peff([peff_m0, peff_m1, peff_m2])
            if peff_v3 is not None:
                iwu_v3 = compute_iwu(et_arr, peff_v3, year, month, version_label='v3_current_prev2')
                out_path_v3 = out_v3 / f'IWU_{year}_{month}.tif'
                write_array_to_raster(iwu_v3, ras_file, ras_file.transform, out_path_v3, 
                                      nodata=no_data_value)

            else:
                logger.warning(f'Skipping v3 for year={year}, month={month} — no valid Peff.')

    logger.info('Monthly IWU calculation completed for all versions.')
    logger.info('---------------------------------------------------------------')


def estimate_growing_season_IWU(years_list, irrigated_cropET_gs_dir, peff_gs_dir,
                                peff_water_year_dir, iwu_output_dir, 
                                skip_processing=False):
    """
    Estimate growing season Irrigation Water Use (IWU) by subtracting effective
    precipitation (Peff, USDA-SCS method) from growing season irrigated crop ET. 
    Two versions are computed based on which Peff accumulation period is used:

        Version 1 — growing season Peff:
            IWU = ET_gs - Peff_gs
            Uses Peff accumulated over the dynamic growing season (same window as ET).
            More physically consistent pairing of ET and Peff.

        Version 2 — water year Peff:
            IWU = ET_gs - Peff_wy
            Uses Peff accumulated over the full water year (Oct–Sep).
            Accounts for carry-over soil moisture from outside the growing season.

    Note:
    -------
    - IWU is only computed where both ET and Peff are valid (not nodata) and ET > 0.


    Output folder structure:
        iwu_output_dir/
        └── IWU_gs/
            ├── IWU_peff_gs/    (Version 1: growing season Peff)
            └── IWU_peff_wy/    (Version 2: water year Peff)

    :param years_list: List/tuple of years to process.
    :param irrigated_cropET_gs_dir: Directory of growing season irrigated crop ET rasters.
    :param peff_gs_dir: Directory of growing season Peff rasters.
    :param peff_water_year_dir: Directory of water year Peff rasters.
    :param iwu_output_dir: Root output directory. Subfolders are created automatically.
    :param skip_processing: Set True to skip this step.

    :return: None.
    """
    if skip_processing:
        return

    # -------------------------------------------------------------------------
    # directory setup
    # -------------------------------------------------------------------------
    iwu_output_dir          = Path(iwu_output_dir)
    irrigated_cropET_gs_dir = Path(irrigated_cropET_gs_dir)
    peff_gs_dir             = Path(peff_gs_dir)
    peff_water_year_dir     = Path(peff_water_year_dir)

    iwu_out_dir_v1 = iwu_output_dir / 'IWU_gs' / 'IWU_peff_gs'
    iwu_out_dir_v2 = iwu_output_dir / 'IWU_gs' / 'IWU_peff_wy'

    for d in [iwu_out_dir_v1, iwu_out_dir_v2]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # helper: load ET, Peff_gs, and Peff_wy arrays for a given year
    # -------------------------------------------------------------------------
    def load_data(year):
        """
        Locate and load growing season ET, Peff_gs, and Peff_wy rasters for a year.
        Returns (et_arr, peff_gs_arr, peff_wy_arr, ras_file), or
        (None, None, None, None) if any input file is missing.
        """
        et_matches      = list(irrigated_cropET_gs_dir.glob(f'*{year}*.tif'))
        peff_gs_matches = list(peff_gs_dir.glob(f'*{year}*.tif'))
        peff_wy_matches = list(peff_water_year_dir.glob(f'*{year}*.tif'))

        if not et_matches or not peff_gs_matches or not peff_wy_matches:
            logger.warning(f'Missing data for year={year} — skipping IWU estimation.')
            return None, None, None, None

        et_arr, ras_file = read_raster_arr_object(et_matches[0], get_file=True)
        peff_gs_arr      = read_raster_arr_object(peff_gs_matches[0], get_file=False)
        peff_wy_arr      = read_raster_arr_object(peff_wy_matches[0], get_file=False)

        return et_arr, peff_gs_arr, peff_wy_arr, ras_file

    # -------------------------------------------------------------------------
    # helper: subtract Peff from ET with validity mask and negative value check
    # -------------------------------------------------------------------------
    def compute_iwu(et_arr, peff_arr):
        """
        Compute IWU = ET - Peff where both inputs are valid (not nodata).
  
        Returns IWU array.
        """
        et_f   = et_arr.astype(np.float32)
        peff_f = peff_arr.astype(np.float32)

        # valid mask: both ET and Peff must be non-nodata
        valid = (et_f != no_data_value) & (et_f > 0) & (peff_f != no_data_value)

        iwu = np.full_like(et_f, no_data_value, dtype=np.float32)
        iwu[valid] = et_f[valid] - peff_f[valid]
        iwu[valid & (iwu < 0)] = 0  # set negative IWU to zero as it is not physically meaningful (precip suffices crop water demand)
        
        return iwu

    # -------------------------------------------------------------------------
    # main processing loop
    # -------------------------------------------------------------------------
    for year in years_list:
        logger.info(f'Estimating growing season IWU for year={year}...')

        et_arr, peff_gs_arr, peff_wy_arr, ras_file = load_data(year)

        # skip year if any input file was missing
        if et_arr is None:
            continue

        # Version 1: IWU = growing season ET - growing season Peff
        iwu_arr_v1 = compute_iwu(et_arr, peff_gs_arr)
        write_array_to_raster(iwu_arr_v1, ras_file, ras_file.transform,
                              iwu_out_dir_v1 / f'IWU_{year}.tif',
                              nodata=no_data_value)

        # Version 2: IWU = growing season ET - water year Peff
        iwu_arr_v2 = compute_iwu(et_arr, peff_wy_arr)
        write_array_to_raster(iwu_arr_v2, ras_file, ras_file.transform,
                              iwu_out_dir_v2 / f'IWU_{year}.tif',
                              nodata=no_data_value)

    logger.info('Growing season IWU calculation completed for all versions.')
    logger.info('---------------------------------------------------------------')


def calculate_irrigated_area_raster(years, irrigated_fraction_dir, irrigated_cropland_dir,
                                    irrigated_area_output_dir, area_unit='hectares',
                                    skip_processing=False):
    """
    Calculate irrigated area (hectares or acres) per pixel by combining irrigated
    fraction data with irrigated cropland classification.

    The irrigated area at each pixel is computed as:

        irrigated_area = irr_fraction * pixel_area

    Pixel area is derived from the raster's geographic CRS (lat/lon degrees) using
    the mid-latitude of the raster extent for representative metre-per-degree conversion.
    The irrigated cropland classification acts as a binary mask — area is only computed
    for pixels classified as irrigated cropland (value = 1).

    Output filename pattern: Irrigated_area_{area_unit}_{year}.tif

    :param years: List/tuple of years to process.
    :param irrigated_fraction_dir: Directory of irrigated fraction rasters (values 0-1).
    :param irrigated_cropland_dir: Directory of irrigated cropland classification rasters
                                    (binary: 1 = irrigated, -9999 = nodata/not irrigated).
    :param irrigated_area_output_dir: Output directory for irrigated area rasters.
    :param area_unit: Unit for output area values. Either 'hectares' (default) or 'acres'.
    :param skip_processing: Set True to skip this step.

    :return: None.
    """
    if skip_processing:
        return

    # -------------------------------------------------------------------------
    # directory setup
    # -------------------------------------------------------------------------
    irrigated_fraction_dir    = Path(irrigated_fraction_dir)
    irrigated_cropland_dir    = Path(irrigated_cropland_dir)
    irrigated_area_output_dir = Path(irrigated_area_output_dir)
    irrigated_area_output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # unit conversion factors from m²
    # -------------------------------------------------------------------------
    conversion = {'hectares': 1 / 10_000,      # 1 ha = 10,000 m²
                  'acres':    1 / 4_046.856}    # 1 acre = 4,046.856 m²

    if area_unit not in conversion:
        raise ValueError(f"area_unit must be 'hectares' or 'acres', got '{area_unit}'.")

    # -------------------------------------------------------------------------
    # compute pixel area from reference raster
    # raster is in geographic CRS (degrees), so pixel area varies with latitude
    # use mid-latitude of the raster extent as a representative value
    # -------------------------------------------------------------------------
    with rio.open(next(irrigated_cropland_dir.glob('*.tif'))) as src:
        transform = src.transform
        bounds    = src.bounds   # (left, bottom, right, top)

    min_lat = bounds.bottom
    max_lat = bounds.top
    mid_lat = (min_lat + max_lat) / 2

    # metres per degree at mid-latitude
    lat_meters_per_degree = 111_320                                         # constant
    lon_meters_per_degree = 111_320 * np.cos(np.radians(mid_lat))          # varies with latitude

    # pixel dimensions in metres
    # a - pixel width in degrees (transform.a)
    # e - pixel height in degrees (transform.e; negative value)
    pixel_height_m = abs(transform.e) * lat_meters_per_degree
    pixel_width_m  = abs(transform.a) * lon_meters_per_degree
    pixel_area_m2  = pixel_height_m * pixel_width_m

    # convert to target unit
    pixel_area = pixel_area_m2 * conversion[area_unit]

    logger.info(f'Mid-latitude of raster extent: {mid_lat:.2f}°N')
    logger.info(f'Pixel area: {pixel_area_m2:.1f} m²  →  {pixel_area:.4f} {area_unit} per pixel')

    # -------------------------------------------------------------------------
    # main processing loop
    # -------------------------------------------------------------------------
    for year in years:
        logger.info(f'Calculating irrigated area ({area_unit}) for year={year}...')

        # locate input files
        frac_matches  = list(irrigated_fraction_dir.glob(f'*{year}*.tif'))
        class_matches = list(irrigated_cropland_dir.glob(f'*{year}*.tif'))

        if not frac_matches or not class_matches:
            logger.warning(f'Missing data for year={year} — skipping.')
            continue

        # load irrigated fraction (0–1) and cropland classification (1 or nodata)
        irr_frac_arr, ras_file = read_raster_arr_object(frac_matches[0], get_file=True)
        irr_class_arr          = read_raster_arr_object(class_matches[0], get_file=False)

        irr_frac_arr  = irr_frac_arr.astype(np.float32)
        irr_class_arr = irr_class_arr.astype(np.float32)

        # valid mask: pixel must be classified as irrigated cropland AND have valid fraction
        valid = (irr_class_arr == 1) & (irr_frac_arr != no_data_value) & (irr_frac_arr > 0) & (irr_frac_arr <= 1)

        # irrigated area = fraction * pixel area, only over classified irrigated pixels
        irrigated_area_arr = np.full_like(irr_frac_arr, no_data_value, dtype=np.float32)
        irrigated_area_arr[valid] = irr_frac_arr[valid] * pixel_area

        # save output
        output_path = irrigated_area_output_dir / f'Irrigated_area_{area_unit}_{year}.tif'
        write_array_to_raster(irrigated_area_arr, ras_file, ras_file.transform,
                              output_path, nodata=no_data_value)

    logger.info(f'Irrigated area ({area_unit}) calculation completed.')
    logger.info('---------------------------------------------------------------')
    
    
def create_spatial_unit_rasters(aquifer_state_shp, raster_config_list,
                                output_dir, ref_raster=WestUS_raster,
                                skip_processing=False):
    """
    Create integer unit ID rasters from an aquifer-state shapefile. 
    Also creates a stateID raster. Multiple aquifer-related rasters 
    can be created in one call by passing multiple configurations.

    Output files:
        output_dir/{raster_name}.tif          — integer ID raster (per config)


    Example raster_configs (using aquifers_by_state.shp attribute names):
    ----------------------------------------------------------------------
        raster_config_list = [
            {
                'raster_name'    : 'aquifer_state_ID',
                'id_attribute'   : 'AQ_ST_ID',   # integer
            },
            {
                'raster_name'    : 'aquifer_ID',
                'id_attribute'   : 'AQ_ID',      # integer
            },
            {
                'raster_name'    : 'state_ID',
                'id_attribute'   : 'State_ID',   # integer
            }
        ]

    :param aquifer_state_shp: Path to aquifer-state shapefile (aquifers_by_state.shp).
                               Must contain all id and name attributes listed in raster_configs.
    :param raster_config_list: List of dicts, each with keys:
                            'raster_name'    — output filename stem (without extension)
                            'id_attribute'   — shapefile integer ID attribute to burn into raster
    :param output_dir: Directory to save output rasters and lookup CSVs.
    :param ref_raster: Reference raster for extent, resolution, and CRS alignment.
                       Default: Western US 2km reference raster.
    :param skip_processing: Set True to skip this step.

    :return: None. Rasters are saved to disk.
    """
    if skip_processing:
        return None

    output_dir        = Path(output_dir)
    aquifer_state_shp = Path(aquifer_state_shp)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load shapefiles
    gdf = gpd.read_file(aquifer_state_shp)

    # create aquifer-related + state ID rasters from raster_configs
    for cfg in raster_config_list:
        if 'raster_name' not in cfg or 'id_attribute' not in cfg:
            raise ValueError('Each raster config must contain "raster_name" and "id_attribute" keys.')
        
        raster_name    = cfg['raster_name']
        id_attribute   = cfg['id_attribute']

        # validate attributes exist in shapefile
        if id_attribute not in gdf.columns:
            raise ValueError(
                f'Attribute {id_attribute} not found in shapefile. '
                f'Available columns: {list(gdf.columns)}'
            )

        logger.info(f'Creating {raster_name}.tif from attribute "{id_attribute}"...')

        # burn integer ID attribute to raster
        shapefile_to_raster(
            input_shape=aquifer_state_shp,
            output_dir=output_dir,
            raster_name=f'{raster_name}.tif',
            burnvalue=None,
            use_attr=True,
            attribute=id_attribute,
            add=None,
            ref_raster=ref_raster,
            resolution=model_res,
            alltouched=False
        )

    logger.info('All spatial unit rasters created.')
    logger.info('---------------------------------------------------------------')


def apply_ref_mask_to_precip(
        precip_monthly_dir,
        ref_raster_path,
        output_dir,
        no_data_value=-9999,
        skip_processing=False):
    """
    Loop over monthly precipitation rasters and set pixel values to nodata
    (-9999) wherever the reference raster has zero values (valid pixel wth Westen US landmass).

    :param precip_monthly_dir: Directory containing monthly precip rasters.
                                Pattern: *.tif
    :param ref_raster_path:    Path to reference raster. Pixels with value == 0
                                are treated as invalid and masked in output.
    :param output_dir:         Directory to save masked output rasters.
    :param no_data_value:      Nodata value written to masked pixels. Default: -9999.
    :param skip_processing:    If True, skip this step.
    """
    if skip_processing:
        return

    precip_monthly_dir = Path(precip_monthly_dir)
    ref_raster_path    = Path(ref_raster_path)
    output_dir         = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load reference raster once — used as mask for all monthly files
    ref_arr  = read_raster_arr_object(ref_raster_path, get_file=False)

    # boolean mask: True where ref == 0 (pixels to be set to nodata)
    zero_mask = (ref_arr == 0)

    # get all monthly precip tifs
    precip_files = sorted(precip_monthly_dir.glob('*.tif'))
    
    if not precip_files:
        logger.warning(f'No .tif files found in {precip_monthly_dir}')
        return
    
    logger.info('-------------------------------------------------------')
    logger.info(f'Masking {len(precip_files)} monthly precip rasters...')

    for fpath in precip_files:
        arr, meta  = read_raster_arr_object(fpath)
        
        # set pixels to nodata where ref raster == 0
        arr[arr == no_data_value] = np.nan
        arr[np.isnan(arr) & zero_mask] = 0

        out_path = output_dir / fpath.name
        write_array_to_raster(arr, meta, meta.transform, out_path, nodata=no_data_value)

    logger.info(f'Done.')
    logger.info('-------------------------------------------------------')



def reclassify_GW_use_perc_rasters(GW_use_perc_dir, westUS_ROI,
                                   output_dir, skip_processing=False):
    """
    Reclassifies Groundwater (GW) use percentage rasters into a binary classification
    (Groundwater-Dominated vs. Conjunctive Use) based on a specific Region of Interest (ROI).

    This function performs the following steps:
    1. Clips the source GW use raster to the West US ROI shapefile.
    2. Manually corrects data for the Rio Grande Basin (San Luis Valley) to 100% GW use
       based on known local irrigation practices.
    3. Reclassifies the data: >= 70% GW use becomes 1 (Dominated), otherwise 0 (Conjunctive).
    4. Masks the result to match the valid pixels of the pumping prediction rasters.

    parameters:

    GW_use_perc_dir (str): Directory containing the source GW use percentage GeoTIFFs.
    westUS_ROI (str): Filepath to the shapefile defining the Western US Region of Interest.
    output_dir (str): Directory path where the output rasters will be saved.
    skip_processing (bool, optional): If True, skips execution. Defaults to False.

    Returns:
        None
    """
    if not skip_processing:

        # making output directories
        GW_use_perc_dir = Path(GW_use_perc_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Opening GW use raster
        # All annual data are same. Taking the 1st one selected.
        gw_use_perc_raster = list(GW_use_perc_dir.glob('*.tif'))[0]
        raster_file = rio.open(gw_use_perc_raster)

        # opening input shapefile and taking it to a GeoJSON format
        shp_extent = gpd.read_file(westUS_ROI)
        shp_extent = shp_extent.to_crs(crs=raster_file.crs)
        geoms = [geom.__geo_interface__ for geom in shp_extent.geometry]  # GeoJSON format

        masked_arr, mask_transform = mask(dataset=raster_file,
                                          shapes=geoms, filled=True,
                                          crop=True, invert=False,
                                          all_touched=False)

        masked_arr = masked_arr.squeeze()

        # saving clipped raster for WestUS ROI
        interim_output_raster = output_dir / 'GW_use_perc_ROI_interim.tif'

        with rio.open(
                interim_output_raster,
                'w',
                driver='GTiff',
                height=masked_arr.shape[0],
                width=masked_arr.shape[1],
                count=1,
                dtype=masked_arr.dtype,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=-9999
        ) as dst:
            dst.write(masked_arr, 1)

        ################################################################################################################
        # # updating GW use % data in the Rio Grande Basin (we know for sure that the northern SLV is
        # ~100% GW irrigated and the southern SLV is ~74% GW irrigated)

        # # northern SLV -> 100%
        lat_min_n, lat_max_n = 37.50, 38.498
        lon_min, lon_max = -106.657, -105.128

        # setting values of the lat-lon window to 100%
        gw_perc = rxr.open_rasterio(interim_output_raster)

        gw_perc.loc[
            dict(
                y=slice(lat_max_n, lat_min_n),
                x=slice(lon_min, lon_max)
            )] = 100
        
        gw_perc_arr = gw_perc.values.squeeze().squeeze()

        # saving the modified raster
        with rio.open(
                os.path.join(output_dir, 'GW_use_perc_ROI.tif'),
                'w',
                driver='GTiff',
                height=gw_perc_arr.shape[0],
                width=gw_perc_arr.shape[1],
                count=1,
                dtype=gw_perc_arr.dtype,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=-9999
        ) as dst:
            dst.write(gw_perc_arr, 1)

        # # southern SLV -> 74%
        lat_min_s, lat_max_s = 36.96, 37.499

        # setting values of the lat-lon window to 74%
        gw_perc_new = rxr.open_rasterio(os.path.join(output_dir, 'GW_use_perc_ROI.tif'))

        gw_perc_new.loc[
            dict(
                y=slice(lat_max_s, lat_min_s),
                x=slice(lon_min, lon_max)
            )] = 74
        
        gw_perc_arr = gw_perc_new.values.squeeze().squeeze()

        # saving the modified raster
        with rio.open(
                os.path.join(output_dir, 'GW_use_perc_ROI_final.tif'),
                'w',
                driver='GTiff',
                height=gw_perc_arr.shape[0],
                width=gw_perc_arr.shape[1],
                count=1,
                dtype=gw_perc_arr.dtype,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=-9999
        ) as dst:
            dst.write(gw_perc_arr, 1)

        ################################################################################################################
        # # creating a classification raster
        # < 30% GW use will be 0 (surface water)
        # 30-70% GW use will be 1 (conjunctive/mixed use)
        # >=70% GW use will be 2 (groundwater-dominated)
        gw_perc_binary = np.where(gw_perc_arr >= 70, 2, np.where(gw_perc_arr >= 30, 1, 0)).squeeze()
        gw_perc_binary = np.where(gw_perc_arr == -9999, -9999, gw_perc_binary)  # preserve nodata values

        # save the binary classified raster
        binary_raster = os.path.join(output_dir, 'Water_source_classification.tif')
        with rio.open(
                binary_raster,
                'w',
                driver='GTiff',
                height=gw_perc_binary.shape[0],
                width=gw_perc_binary.shape[1],
                count=1,
                dtype=gw_perc_binary.dtype,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=-9999
        ) as dst:
            dst.write(gw_perc_binary, 1)

        # delete the interim raster
        gw_perc.close()
        os.remove(interim_output_raster)


def merge_ORNl_Dayflow_annual_data(input_annual_csv_dir, skip_processing=False):
    """
    Merge per-year ORNL Dayflow CSV files into a single dataset and convert
    streamflow units from cfs to area-normalized mm.

    Parameters
    ----------
    input_annual_csv_dir : str or Path
        Directory containing per-year Dayflow CSV files
        (e.g. ``Data_main/rasters/Dayflow/processed``).
        Files whose names contain 'cache' are skipped.
    skip_processing : bool, optional
        If True, returns immediately without doing anything. Default False.

    Returns
    -------
    None
    
    Output is written as a single merged CSV file.
    """
    if skip_processing:
        return
    
    print('Merging ORNL Dayflow annual CSV files...')
    
    # get all annual csv files
    input_annual_csv_list = list(Path(input_annual_csv_dir).glob('*.csv'))
    input_annual_csv_list = [i for i in input_annual_csv_list if 'cache' not in str(i)]
    
    if not input_annual_csv_list:
        logger.warning(f'No csv files fround in {input_annual_csv_dir}')
        
    # read and concatenate all annual csvs into one dataframe
    df_list = []
    
    for csv_file in input_annual_csv_list:
        df = pd.read_csv(csv_file)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.rename(columns={'Sim_Q_N': 'Sim_Q_naturalized',
                                          'Sim_Q_C': 'Sim_Q_assimilated'})   # units in cfs
    
    # converting unit from 'cfs' to area-normalized 'mm'
    days_in_month = {1: 31, 
                     2: 28, 
                     3: 31, 
                     4: 30,
                     5: 31, 
                     6: 30, 
                     7: 31, 
                     8: 31, 
                     9: 30, 
                     10: 31, 
                     11: 30, 
                     12: 31}
    
    days = merged_df['Month'].map(days_in_month)
    merged_df['Sim_Q_naturalized_mm'] = merged_df['Sim_Q_naturalized'] * days * 1.9835 / (merged_df['Outlet_Drainage_Area'] * 247.105) * 304.8
    merged_df['Sim_Q_assimilated_mm'] = merged_df['Sim_Q_assimilated'] * days * 1.9835 / (merged_df['Outlet_Drainage_Area'] * 247.105) * 304.8


    # HUC8 columns converted to string and 8 digits with leading zeros (e.g., '01010001')
    merged_df['HUC8'] = merged_df['HUC8'].astype(str).str.zfill(8)
    
    # save merged dataframe to a new csv
    output_csv_path = Path(input_annual_csv_dir).parent / 'Merged_Dayflow.csv'

    merged_df.to_csv(output_csv_path, index=False)
    


def build_processed_huc8(huc8_shp, states_shp, aquifer_shp, irr_cropland_datadir, 
                         years, output_shp, irr_classification_threshold=3, skip_processing=False):
    """
    Build a GeoDataFrame of western US HUC8 watersheds with state,
    aquifer, and irrigated-area attributes.

    Workflow
    --------
    1. Assigns each HUC8 to the state it overlaps most (by area).
    2. Assigns each HUC8 to the aquifer region it overlaps most (by area).
       HUC8s outside any aquifer boundary retain NaN for aquifer columns.
    3. Computes the median annual irrigated pixel count per HUC8 across all
       years using vectorized zonal statistics, then converts to km².
    4. Flags each HUC8 as irrigated if irrigated area exceeds {irr_classification_threshold}% of total area.

    Parameters
    ----------
    huc8_shp : str or Path
        Path to the HUC8 watershed shapefile.
    states_shp : str or Path
        Path to the western US states shapefile.
    aquifer_shp : str or Path
        Path to the aquifer-region shapefile (aquifers_by_state.shp).
    irr_cropland_datadir : str or Path
        Directory containing annual irrigated-cropland rasters
        (one .tif per year, filename must contain the year).
    years : list of int
        Years to include when computing median irrigated area
        (e.g. list(range(1986, 2024))).

    Returns
    -------
    None. 
    
    Saves a processed HUC8 shapefile with added attributes for state, aquifer region, area, 
    median irrigated area, and irrigated classification.
    """

    if skip_processing:
        return None

    huc8 = gpd.read_file(huc8_shp)
    states = gpd.read_file(states_shp)
    
    print('Processing HUC8 shapefile to bring state and major aquifer boundary information + \n',
    'classifyinhg each HUC8 as irrigated or non-irrigated based on median irrigated area across 1986-2023\n...')

    ############### Ovelap HUC8 with states to assign state name to each HUC8 ###############

    huc8 = huc8.to_crs('EPSG:5070')  # Reproject states to projected crs
    states = states.to_crs('EPSG:5070')  # Reproject states to projected crs

    # Intersection - creates new polygons where the shapefiles overlap
    overlap = gpd.overlay(huc8, states, how='intersection')

    # calculate the area of each intersection polygon
    overlap['overlap_area'] = overlap.geometry.area

    # for each HUC8, find the state with the largest overlap area
    idx = overlap.groupby('HUC8')['overlap_area'].idxmax()
    huc8_state_overlap = overlap.iloc[idx][['HUC8', 'NAME_1', 'NAME_2']]
    overlap_state = huc8.merge(huc8_state_overlap, on='HUC8', how='left')

    # keeping selected columns
    overlap_state.rename(columns={'NAME_1': 'HUC8_name', 'NAME_2': 'State'}, inplace=True)

    ############### Ovelap HUC8 with aquifer-region to assign these aquifer boundaries to each HUC8 ###############

    aquifer_region = gpd.read_file(aquifer_shp)
    aquifer_region = aquifer_region.to_crs('EPSG:5070')  # Reproject to projected crs

    # Step 1: Intersection to find which aquifer each HUC8 overlaps with (and by how much)
    overlap_2 = gpd.overlay(overlap_state, aquifer_region, how='intersection', keep_geom_type=False)
    overlap_2['overlap_area'] = overlap_2.geometry.area

    # For each HUC8, find the aquifer_region with the largest overlap area
    idx = overlap_2.groupby('HUC8')['overlap_area'].idxmax()
    best_aquifer_match = overlap_2.iloc[idx][['HUC8', 'ROCK_NAME', 'AQ_NAME', 'AQ_code', 'State_code', 'AQ_State', 'AQ_Region']]

    # Step 2: Left-merge back to overlap_state so ALL ~1187 HUC8s are retained.
    # HUC8s outside any aquifer boundary get NaN for aquifer columns.
    processed_huc8 = overlap_state.merge(best_aquifer_match, on='HUC8', how='left')
    processed_huc8['area_km2'] = processed_huc8.geometry.area / 1e6  # Convert from m^2 to km^2

    print(f"Total HUC8s: {processed_huc8['HUC8'].nunique()}")
    print(f"HUC8s with aquifer assignment: {processed_huc8['AQ_Region'].notna().sum()}")
    print(f"HUC8s without aquifer (outside coverage): {processed_huc8['AQ_Region'].isna().sum()}")

    ############### Median irrigated area per HUC8 (vectorized zonal_stats) ###############

    irr_cropland_datadir = Path(irr_cropland_datadir)

    # Reproject all HUC8s once to match raster CRS
    sample_raster = list(irr_cropland_datadir.glob(f'*{years[0]}*.tif'))[0]
    with rio.open(sample_raster) as src:
        raster_crs = src.crs

    huc8_reproj = processed_huc8.to_crs(raster_crs)  # reproject once, not 45,000 times

    # One zonal_stats call per year across ALL HUC8s
    all_counts = []
    for year in years:
        irr_data = list(irr_cropland_datadir.glob(f'*{year}*.tif'))[0]
        stats = zonal_stats(huc8_reproj, str(irr_data), stats=['count'], nodata=-9999)
        counts = [s['count'] if s['count'] is not None else np.nan for s in stats]
        all_counts.append(counts)

    # all_counts shape: (n_years x n_huc8s) — median across years per HUC8
    counts_df = pd.DataFrame(all_counts, columns=processed_huc8.index)
    processed_huc8['median_irr_pixel_count'] = counts_df.median(axis=0).values
    processed_huc8['irr_area'] = processed_huc8['median_irr_pixel_count'] * 4  # km²

    irrigated_huc8 = processed_huc8['irr_area'] > (irr_classification_threshold * processed_huc8['area_km2'] / 100)  # irrigated HUC8 mask
    processed_huc8['Irrigated'] = irrigated_huc8

    # Save the processed HUC8 geodataframe to a shapefile
    processed_huc8.to_file(output_shp, driver='ESRI Shapefile')

    print('\nProcessed HUC8 shapefile saved to ->', output_shp)


def develop_P_PET_correlation_dataset(years_to_consider, monthly_precip_dir, monthly_pet_dir,
                                      output_dir, skip_processing=False):
    """
    Develop PET and P correlation dataset (static) for the Western US.

    :param years_to_consider: List of years to consider. 
                              Only months from these years will be included 
                              in the correlation calculation.
    :param monthly_precip_dir: Filepath of monthly precip directory.
    :param monthly_pet_dir: Filepath of monthly pet directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip creating this dataset.

    :return: None
    """
    if skip_processing:
        return None

    print('creating P-PET correlation dataset...')

    makedirs([output_dir])

    # accumulating precip and pet data
    monthly_precip_dir = Path(monthly_precip_dir)
    monthly_pet_dir = Path(monthly_pet_dir)
    
    # Sort both lists by filename to ensure temporal alignment
    monthly_precip_data_list = sorted(
        list(monthly_precip_dir.glob('*.tif')),
        key=lambda x: x.name  # assumes filenames sort chronologically
    )
    monthly_pet_data_list = sorted(
        list(monthly_pet_dir.glob('*.tif')),
        key=lambda x: x.name
    )
    
    monthly_precip_data_list = [f for f in monthly_precip_data_list if any(str(year) in f.name for year in years_to_consider)]
    monthly_pet_data_list = [f for f in monthly_pet_data_list if any(str(year) in f.name for year in years_to_consider)]
    
    # Verify count match
    assert len(monthly_precip_data_list) == len(monthly_pet_data_list), \
        f"File count mismatch! Precip: {len(monthly_precip_data_list)}, PET: {len(monthly_pet_data_list)}"

    # Verify temporal alignment of pairs
    print('Verifying temporal alignment...')
    for p, pet in zip(monthly_precip_data_list, monthly_pet_data_list):
        print(f'  P: {p.name} | PET: {pet.name}')
    
    # reading datasets as arrays
    monthly_precip_arr_list = [read_raster_arr_object(i, get_file=False) for i in monthly_precip_data_list]
    monthly_pet_arr_list = [read_raster_arr_object(i, get_file=False) for i in monthly_pet_data_list]

    # stacking monthly datasets into a list
    precip_stack = np.stack(monthly_precip_arr_list,
                            axis=0)  # shape becomes - n_months, n_lat (height), n_lon(width)
    pet_stack = np.stack(monthly_pet_arr_list, axis=0)  # shape becomes - n_months, n_lat (height), n_lon(width)

    # replacing nodata with nan so nanmean/nansum ignore them correctly
    precip_stack = precip_stack.astype(float)
    pet_stack = pet_stack.astype(float)
    precip_stack[precip_stack == no_data_value] = np.nan
    pet_stack[pet_stack == no_data_value] = np.nan

    # Calculating mean along the time axis (i.e., across months) for each pixel
    precip_mean = np.nanmean(precip_stack, axis=0)
    pet_mean = np.nanmean(pet_stack, axis=0)

    # estimating precip and pet anomalies
    precip_anomalies = precip_stack - precip_mean
    pet_anomalies = pet_stack - pet_mean

    # getting numerator (covariance) for each pixel across time
    numerator = np.nansum(precip_anomalies * pet_anomalies, axis=0)

    # getting denominator (sum of squares for both variables (this measures the total variation for each))
    sum_of_squares_precip = np.sqrt(np.nansum(precip_anomalies ** 2, axis=0))
    sum_of_squares_pet = np.sqrt(np.nansum(pet_anomalies ** 2, axis=0))
    denominator = sum_of_squares_precip * sum_of_squares_pet

    # calculating Pearson correlation for each pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_arr = numerator / denominator

    output_raster = os.path.join(output_dir, 'PET_P_corr.tif')
    _, ref_file = read_raster_arr_object(monthly_precip_data_list[0])
    write_array_to_raster(correlation_arr, ref_file, ref_file.transform, output_raster)


def calculate_growing_season_precip_fraction(years_to_consider, monthly_precip_dir,
                                             output_dir, skip_processing=False):
    """
    Calculate the fraction (%) of April–October precipitation out of annual precipitation
    for each pixel, saved as one raster per year.

    :param years_to_consider: List of years to process.
    :param monthly_precip_dir: Filepath of monthly precip directory.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this step.

    :return: None
    """
    if skip_processing:
        return None

    print('calculating growing season precip fraction (Apr-Oct / Annual)...')

    makedirs([output_dir])

    monthly_precip_dir = Path(monthly_precip_dir)
    growing_season_months = list(range(4, 11))  # April (4) to October (10)

    for year in years_to_consider:
        # collect all 12 months for this year
        annual_files = list(monthly_precip_dir.glob(f'*{year}*.tif'))

        if len(annual_files) == 0:
            logger.warning(f'No monthly precip files found for {year}, skipping.')
            continue

        # growing season months only
        gs_files = [f for f in annual_files
                    if int(f.name.split('.tif')[0].split('_')[-1]) in growing_season_months]

        annual_stack = np.stack(
            [read_raster_arr_object(f, get_file=False).astype(float) for f in annual_files], axis=0
        )
        gs_stack = np.stack(
            [read_raster_arr_object(f, get_file=False).astype(float) for f in gs_files], axis=0
        )

        annual_stack[annual_stack == no_data_value] = np.nan
        gs_stack[gs_stack == no_data_value] = np.nan

        annual_precip = np.nansum(annual_stack, axis=0)
        gs_precip = np.nansum(gs_stack, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            gs_precip_fraction = np.where(annual_precip > 0,
                                          (gs_precip / annual_precip) * 100,
                                          np.nan)

        _, ref_file = read_raster_arr_object(annual_files[0])
        output_raster = os.path.join(output_dir, f'GS_precip_fraction_{year}.tif')
        write_array_to_raster(gs_precip_fraction, ref_file, ref_file.transform, output_raster)

        logger.info(f'Growing season precip fraction for {year} saved to {output_raster}')


def run_all_preprocessing(years_list,
                          skip_process_GrowSeason_data=False,
                          skip_ref_mask_prism_precip=False,
                          skip_sum_precip_growing_season=False,
                          skip_sum_winter_precip=False,
                          skip_prism_tmean_processing=False,
                          skip_irr_cropET_data_merge=False,
                          skip_sum_irrigated_cropET=False,
                          skip_sum_usda_scs_peff_growing_season=False,
                          skip_sum_usda_scs_peff_water_year=False,
                          skip_merge_irr_fraction_data=False,
                          skip_irr_cropland_classification=False,
                          skip_estimate_irrigated_area=False,
                          skip_calculate_monthly_IWU=False,
                          skip_calculate_growing_season_IWU=False,
                          skip_create_water_source_rasters=False,
                          skip_merge_ORNl_Dayflow_annual_data=False,
                          skip_build_processed_huc8=False,
                          skip_develop_P_PET_correlation_dataset=False,
                          ):
    """
    Run all data pre-processing steps.
    """

    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # apply reference mask to monthly precipitation rasters
    apply_ref_mask_to_precip(
        precip_monthly_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly',
        ref_raster_path=WestUS_raster,
        output_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked',
        skip_processing=skip_ref_mask_prism_precip)
    
    # PRISM precipitation data processing (growing season sum)
    dynamic_gs_sum_of_variable(year_list=years_list,
                               growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                               monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked',
                               gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/growing_season',
                               ref_raster=WestUS_raster,
                               sum_keyword='Precip', skip_processing=skip_sum_precip_growing_season)
    
    # Sum precipitation for winter months
    sum_precip_winter_months(years_list=years_list, 
                             var_monthly_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked', 
                             output_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/winter', 
                             save_keyword='WinterPrecip', 
                             skip_processing=skip_sum_winter_precip)

    # PRISM temperature (mean) data processing (growing season average)
    dynamic_gs_mean_of_variable(year_list=years_list,   
                                growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                                monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Tmean/monthly',
                                gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Tmean/growing_season',
                                mean_keyword='Tmean', skip_processing=skip_prism_tmean_processing)

    # Join (merge) irrigated cropET data chunks to Western US extent (1986-2024)
    merge_GEE_data_patches_IrrMapper_LANID_extents(
        year_with_full_extent=years_list,       
        year_with_partial_extent=None,
        input_dir_irrmapper=PROJECT_ROOT / 'Data_main/rasters/Irrig_crop_OpenET_IrrMapper',
        input_dir_lanid=PROJECT_ROOT / 'Data_main/rasters/Irrig_crop_OpenET_LANID',
        merged_output_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly',
        merge_keyword='Irrigated_cropET', monthly_data=True,
        ref_raster=WestUS_raster,
        skip_processing=skip_irr_cropET_data_merge)

    # Sum irrigated crop ET for dynamic growing season
    dynamic_gs_sum_of_variable(year_list=years_list,
                               growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                               monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly',
                               gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/growing_season',
                               sum_keyword='Irrigated_cropET',
                               skip_processing=skip_sum_irrigated_cropET)
    
    # Sum effective precipitation for dynamic growing season
    dynamic_gs_sum_of_variable(year_list=years_list,
                               growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                               monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/monthly',
                               gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/growing_season',
                               sum_keyword='effective_precip',
                               skip_processing=skip_sum_usda_scs_peff_growing_season)
    
    # Sum effective precipitation for water year
    sum_vars_water_yr(years_list=years_list, 
                      var_monthly_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/monthly', 
                      output_dir_water_yr=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/water_year',
                      save_keyword='effective_precip', skip_processing=skip_sum_usda_scs_peff_water_year)

    # process irrigated fraction data (1986-2024)
    merge_GEE_data_patches_IrrMapper_LANID_extents(
        year_with_full_extent=years_list,
        year_with_partial_extent=None,
        input_dir_irrmapper=PROJECT_ROOT / 'Data_main/rasters/Irrigation_Frac_IrrMapper',
        input_dir_lanid=PROJECT_ROOT / 'Data_main/rasters/Irrigation_Frac_LANID',
        merged_output_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
        merge_keyword='Irrigated_Frac', monthly_data=False,
        ref_raster=WestUS_raster,
        skip_processing=skip_merge_irr_fraction_data)

    # process irrigated cropland data (1986-2024)
    classify_irrigated_cropland(years=years_list,
                                irrigated_fraction_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
                                irrigated_cropland_output_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland',
                                basin_range_shp=PROJECT_ROOT / 'Data_main/shapefiles/Basin_Range_aquifer/Basin_RangeFill_extent.shp',
                                skip_processing=skip_irr_cropland_classification)
    
    # calculate irrigated area (hectares) by combining irrigated fraction and cropland classification
    calculate_irrigated_area_raster(years=years_list, 
                                    irrigated_fraction_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
                                    irrigated_cropland_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland',
                                    irrigated_area_output_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_area',
                                    area_unit='hectares',
                                    skip_processing=skip_estimate_irrigated_area)
    
    # calculate monthly IWU
    calculate_monthly_IWU(years_list=years_list,
                          irrigated_cropET_monthly_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly',
                          peff_monthly_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/monthly',
                          iwu_output_dir=PROJECT_ROOT / 'Data_main/rasters/IWU',
                          skip_processing=skip_calculate_monthly_IWU)
    
    # calculate growing season IWU
    estimate_growing_season_IWU(years_list=years_list, 
                                irrigated_cropET_gs_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/growing_season',
                                peff_gs_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/growing_season',
                                peff_water_year_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/water_year',
                                iwu_output_dir=PROJECT_ROOT / 'Data_main/rasters/IWU',
                                skip_processing=skip_calculate_growing_season_IWU)
    
    # reclassify GW use percentage rasters into binary classification (GW-dominated vs. conjunctive use)
    reclassify_GW_use_perc_rasters(GW_use_perc_dir=PROJECT_ROOT / 'Data_main/rasters/USGS_GW_%',
                                   westUS_ROI=WestUS_shape,
                                   output_dir=PROJECT_ROOT / 'Data_main/rasters/USGS_GW_%/Water_source_classification',
                                   skip_processing=skip_create_water_source_rasters)
    
    # merge ORNL Dayflow annual data into one csv
    merge_ORNl_Dayflow_annual_data(input_annual_csv_dir=PROJECT_ROOT / 'Data_main/rasters/Dayflow/processed', 
                                   skip_processing=skip_merge_ORNl_Dayflow_annual_data) 


    # build processed HUC8 geodataframe with state and aquifer info + irrigated/non-irrigated classification
    build_processed_huc8(
    huc8_shp=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8.shp',
    states_shp=PROJECT_ROOT / "Data_main/ref_shapes/WestUS_states.shp",
    aquifer_shp=PROJECT_ROOT / "Data_main/ref_shapes/aquifers_ROI/aquifers_by_state.shp",
    irr_cropland_datadir=PROJECT_ROOT / "Data_main/rasters/Irrigated_cropland",
    output_shp=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8_processed.shp',
    irr_classification_threshold=3,  # HUC8 classified as irrigated if median irrigated area > 3% of total area
    years=list(range(1986, 2024)),
    skip_processing=skip_build_processed_huc8)
    
    
    # develop a static raster of pixel-wise Pearson correlation between monthly precip and PET anomalies across 1986-2023
    develop_P_PET_correlation_dataset(years_to_consider=list(range(1986, 2024)),
                                      monthly_precip_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked',
                                      monthly_pet_dir=PROJECT_ROOT / 'Data_main/rasters/GRIDMET_RET/monthly',
                                      output_dir=PROJECT_ROOT / 'Data_main/rasters/PET_P_correlation',
                                      skip_processing=skip_develop_P_PET_correlation_dataset)

