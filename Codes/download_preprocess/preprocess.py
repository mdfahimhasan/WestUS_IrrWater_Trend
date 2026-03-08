# author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import re
import sys
import logging
import datetime
import numpy as np
import rasterio as rio
from pathlib import Path
from rasterio.warp import reproject, Resampling

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, sum_rasters, write_array_to_raster, \
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

    if not skip_processing:

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
    :param input_cropET_monthly_dir: Directory file path of monthly datasets of the variable of interest.
    :param output_dir_water_yr: File path of directory to save summed variable for each water year.
    :param save_keyword: Keyword to use for summed cropET data saving.
    :param skip_processing: Set True to skip processing.

    :return: None.
    """
    if not skip_processing:
        output_dir_water_yr = Path(output_dir_water_yr)
        output_dir_water_yr.mkdir(parents=True, exist_ok=True)

        for yr in years_list:
            print(f'summing monthly cropET for water year {yr}...')

            # summing rainfed/irrigated crop ET for water year (previous year's October to current year's september)
            et_data_prev_years = var_monthly_dir.glob(f'*{yr - 1}_1[0-2].*tif')
            et_data_current_years = var_monthly_dir.glob(f'*{yr}_[1-9].*tif')
            et_water_yr_list = et_data_prev_years + et_data_current_years

            sum_rasters(raster_list=et_water_yr_list, raster_dir=None,
                        output_raster=output_dir_water_yr / f'{save_keyword}_{yr}.tif',
                        ref_raster=et_water_yr_list[0])
    else:
        pass


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
    if not skip_processing:
        monthly_input_dir = Path(monthly_input_dir)
        gs_output_dir = Path(gs_output_dir)
        gs_output_dir.mkdir(parents=True, exist_ok=True)

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            logging.info(f'Dynamically summing {sum_keyword} monthly datasets for growing season {year}...')

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

        logging.info('All dynamic summing completed')
        logging.info('---------------------------------------------------------------')


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
    if not skip_processing:
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

        logging.info('All dynamic averaging completed')
        logging.info('---------------------------------------------------------------')


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
                                irr_fraction_threshold_others,
                                irr_fraction_threshold_BasinRange,
                                basin_range_shp,
                                skip_processing=False):
    """
    Classifies irrigated cropland using irrigated fraction data.

    :param years: List of years to process data for.
    :param irrigated_fraction_dir: Input directory path for irrigated fraction data.
    :param irrigated_cropland_output_dir: Output directory path for classified irrigated cropland data.
    :param irr_fraction_threshold_others: Minimum threshold (float) to consider a pixel irrigated in
                                          regions outside basin and range-fill region.
    :param irr_fraction_threshold_BasinRange: Minimum threshold (float) to consider a pixel irrigated in
                                              regions inside basin and range-fill region.
    :param basin_range_shp: Basin and range-fill region shapefile.
    :param skip_processing: Set True to skip classifying irrigated and rainfed cropland data.

    :return: None
    """
    if not skip_processing:
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

            # classify irrigated cropland for basin and range.  -9999 is no data
            irrigated_cropland = np.where((basin_range_mask == 1) & (irrig_arr > irr_fraction_threshold_BasinRange), 1,
                                          irrigated_cropland)

            # classification using defined irrigated fraction. -9999 is no data
            irrigated_cropland = np.where((basin_range_mask == 0) & (irrig_arr > irr_fraction_threshold_others), 1,
                                          irrigated_cropland)

            # saving classified data
            output_irrigated_cropland_raster = irrigated_cropland_output_dir / f'Irrigated_cropland_{year}.tif'

            write_array_to_raster(raster_arr=irrigated_cropland, raster_file=irrig_file,
                                  transform=irrig_file.transform,
                                  output_path=output_irrigated_cropland_raster,
                                  dtype=np.int32)  # linux can't save data properly if dtype isn't np.int32 in this case
    else:
        pass


def create_stateID_raster(westUS_shp, output_dir, skip_processing=False):
    """
    Create a stateID reference raster.

    :param westUS_shp: Western US shapefile with the attribute 'stateID'.
    :param output_dir: Output directory to save the created raster.
    :param skip_processing: Set True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([output_dir])

        shapefile_to_raster(input_shape=westUS_shp, output_dir=output_dir, raster_name='stateID.tif',
                            burnvalue=None, use_attr=True,
                            attribute='stateID', add=None, ref_raster=WestUS_raster,
                            resolution=model_res, alltouched=False)

        logger.info('created stateID reference raster...')

    else:
        pass


def run_all_preprocessing(years_list,
                          skip_process_GrowSeason_data=False,
                          skip_prism_precip_processing=False,
                          skip_prism_tmean_processing=False,
                          skip_irr_cropET_data_merge=False,
                          skip_sum_irrigated_cropET=False,
                          skip_sum_usda_scs_peff_growing_season=False,
                          skip_sum_usda_scs_peff_water_year=False):
    """
    Run all data pre-processing steps.
    """

    # process growing season data
    extract_month_from_GrowSeason_data(GS_data_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                                       skip_processing=skip_process_GrowSeason_data)

    # PRISM precipitation data processing (growing season sum)
    dynamic_gs_sum_of_variable(year_list=years_list,
                               growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                               monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly',
                               gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/growing_season',
                               ref_raster=WestUS_raster,
                               sum_keyword='Precip', skip_processing=skip_prism_precip_processing)

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
    dynamic_gs_sum_of_variable(year_list=years_list,
                               growing_season_dir=PROJECT_ROOT / 'Data_main/rasters/Growing_season',
                               monthly_input_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/monthly',
                               gs_output_dir=PROJECT_ROOT / 'Data_main/rasters/Peff_usda_scs/water_year',
                               sum_keyword='effective_precip',
                               skip_processing=skip_sum_usda_scs_peff_water_year)

    # # process irrigated fraction data (2021-2023)
    # # processed in the Peff paper
    # merge_GEE_data_patches_IrrMapper_LANID_extents(
    #     year_with_full_extent=(1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    #                            2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
    #                            2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024),
    #     year_with_partial_extent=None,
    #     input_dir_irrmapper='../../Data_main/Rasters/Irrigation_Frac_IrrMapper',
    #     input_dir_lanid='../../Data_main/Rasters/Irrigation_Frac_LANID',
    #     merged_output_dir='../../Data_main/Rasters/Irrigated_cropland/Irrigated_Frac',
    #     merge_keyword='Irrigated_Frac', monthly_data=False,
    #     ref_raster=WestUS_raster,
    #     skip_processing=skip_irr_frac_data_processing)

    # # process irrigated cropland data (2000-2023)
    # # 2000-2020 data was processed int he Peff paper
    # classify_irrigated_cropland(years=list(range(2000, 2024)),
    #                             irrigated_fraction_dir='../../Data_main/rasters/Irrigated_cropland/Irrigated_Frac',
    #                             irrigated_cropland_output_dir='../../Data_main/rasters/Irrigated_cropland',
    #                             irr_fraction_threshold_others=0.13,  # 13%
    #                             irr_fraction_threshold_BasinRange=0.01,  # 1%
    #                             basin_range_shp='../../Data_main/shapefiles/Basin_Range_aquifer/Basin_RangeFill_extent.shp',
    #                             skip_processing=skip_irr_cropland_classification)

