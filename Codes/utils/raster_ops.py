# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
import numpy as np
import rasterio as rio
from osgeo import gdal
import geopandas as gpd
from pathlib import Path
from rasterio import features
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.enums import Resampling


# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_states.shp'
WestUS_raster = PROJECT_ROOT / 'Data_main/ref_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = PROJECT_ROOT / 'Data_main/ref_rasters/GEE_merging_refraster_larger_grids.tif'


def read_raster_arr_object(raster_file, rasterio_obj=False, band=1, get_file=True, change_dtype=True):
    """
    Get raster array and raster file.

    :param raster_file: Input raster filepath.
    :param rasterio_obj: Set True if raster_file is a rasterio object.
    :param band: Selected band to read. Default set to 1.
    :param get_file: Set to False if raster file is not required.
    :param change_dtype: Set True if you want to change raster data type to float. Default set to True.

    :return: Raster numpy array and rasterio object file (get_file=True, rasterio_obj=False).
    """
    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False

    raster_arr = raster_file.read(band)

    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan

    if get_file:
        return raster_arr, raster_file
    else:
        return raster_arr


def write_array_to_raster(raster_arr, raster_file, transform, output_path, dtype=None,
                          ref_file=None, nodata=no_data_value):
    """
    Write raster array to Geotiff format.

    :param raster_arr: Raster array data to be written.
    :param raster_file: Original rasterio raster file containing geo-coordinates.
    :param transform: Affine transformation matrix.
    :param output_path: Output filepath.
    :param dtype: Output raster data type. Default set to None.
    :param ref_file: Write output raster considering parameters from reference raster file.
    :param nodata: no_data_value set as -9999.

    :return: Output filepath.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if dtype is None:
        dtype = raster_arr.dtype

    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform

    with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=dtype,
            count=raster_file.count,
            crs=raster_file.crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    return output_path


def mask_raster_by_shape(input_raster, input_shape, output_dir,
                         raster_name, band_names=None,
                         crop=True, filled=True, invert=False,
                         nodata=no_data_value):
    """
    Crop/mask a raster with a given shapefile/raster's extent.

    :param input_raster: Filepath of input raster.
    :param input_shape: Filepath of shape file to crop input_raster.
    :param output_dir: Filepath of output directory.
    :param raster_name: Masked raster name.
    :param band_names: A list of band names (str) to set with the output raster. Default set to None.
    :param crop: Set to False to set no data in pixels outside shape. Only the pixels inside will have values
                 and the row and column size will be the same as input raster.
                 Default set to True. The pixels outside the shape will be discarded totally and row and column values
                 will change.
    :param invert : If False (default) pixels outside shapes will be masked.  If True,
                    pixels inside shape will be masked.
    :param filled : If True, the pixels outside the features will be set to nodata.
                    If False, the output array will contain the original pixel data,
                    and only the mask will be based on shapes.  Defaults to True.
    :param nodata: No data value. Default set to -9999.

    :return: Filepath of cropped raster.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # opening input raster
    raster_file = rio.open(input_raster)
    raster_arr = raster_file.read()

    # opening input shapefile and taking it to a GeoJSON format
    shp_extent = gpd.read_file(input_shape)
    shp_extent = shp_extent.to_crs(crs=raster_file.crs)
    geoms = [geom.__geo_interface__ for geom in shp_extent.geometry]  # GeoJSON format

    # masking
    masked_arr, mask_transform = mask(dataset=raster_file, shapes=geoms, filled=filled,
                                      crop=crop, invert=invert, all_touched=False)

    # naming output file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_raster = output_dir / raster_name

    # saving output raster
    if band_names is not None:
        with rio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=masked_arr.shape[-2],  # using reverse indexing to handle multi-band rasters
                width=masked_arr.shape[-1],
                dtype=raster_arr.dtype,
                count=raster_file.count,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=nodata
        ) as dst:
            dst.write(masked_arr)
            dst.descriptions = tuple(band_names)

    else:
        with rio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=masked_arr.shape[-2],  # using reverse indexing to handle multi-band rasters
                width=masked_arr.shape[-1],
                dtype=raster_arr.dtype,
                count=raster_file.count,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=nodata
        ) as dst:
            dst.write(masked_arr)

    return output_raster


def set_nodata_inside_shapefile(input_raster, input_shape, output_dir,
                                raster_name, band_names=None,
                                nodata=no_data_value):
    """
    Set nodata to pixels of the raster outside the shapefile. uses rasterio.mask.mask()

    :param input_raster: Filepath of input raster.
    :param input_shape: Filepath of shape file to crop input_raster.
    :param output_dir: Filepath of output directory.
    :param raster_name: Masked raster name.
    :param band_names: A list of band names (str) to set with the output raster. Default set to None.
    :param nodata: No data value. Default set to -9999.

    :return: Filepath of masked raster.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # opening input raster
    raster_file = rio.open(input_raster)
    raster_arr = raster_file.read()

    # opening input shapefile and taking it to a GeoJSON format
    shp_extent = gpd.read_file(input_shape)
    shp_extent = shp_extent.to_crs(crs=raster_file.crs)
    geoms = [geom.__geo_interface__ for geom in shp_extent.geometry]  # GeoJSON format

    # masking
    masked_arr, mask_transform = mask(dataset=raster_file, shapes=geoms, filled=False,
                                      crop=False, invert=True, all_touched=False)

    # naming output file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_raster = output_dir / raster_name

    # saving output raster
    if band_names is not None:
        with rio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=masked_arr.shape[-2],  # using reverse indexing to handle multi-band rasters
                width=masked_arr.shape[-1],
                dtype=raster_arr.dtype,
                count=raster_file.count,
                crs=raster_file.crs,
                transform=mask_transform,
                nodata=nodata
        ) as dst:
            dst.write(masked_arr)
            dst.descriptions = tuple(band_names)

    else:
        with rio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=masked_arr.shape[-2],  # using reverse indexing to handle multi-band rasters
                width=masked_arr.shape[-1],
                dtype=raster_arr.dtype,
                count=raster_file.count,
                crs=raster_file.crs,
                transform=raster_file.transform,
                nodata=nodata
        ) as dst:
            dst.write(masked_arr)

    return output_raster


def mosaic_rasters_from_directory(input_dir, output_dir, raster_name, ref_raster=WestUS_raster, search_by="*.tif",
                                  dtype=None, resampling_method='nearest', mosaicking_method='first',
                                  resolution=None, nodata=no_data_value):
    """
    Mosaics multiple rasters into a single raster from a directory (rasters have to be in the same directory).

    :param input_dir: Input rasters' directory.
    :param output_dir: Output raster directory.
    :param raster_name: Output raster name.
    :param ref_raster: Reference raster filepath. Set default to WestUS_raster.
    :param search_by: Input raster search criteria. Default set to '*.tif'.
    :param dtype: Output raster data type. Default set to None.
    :param resampling_method: Resampling method. Default set to 'nearest'. Can also take 'bilinear'. Currently, can use
                              only these two resampling methods.
    :param mosaicking_method: Mosaicking method. Can be 'first' or 'max' or 'min'. Default set to 'first'.
    :param resolution: Resolution of the output raster. Default set to None to use first input raster's resolution.
    :param nodata: no_data_value set as -9999.

    :return: Mosaicked raster array and filepath of mosaicked raster.
    """
    input_rasters = list(Path(input_dir).glob(search_by))

    raster_list = []
    for raster in input_rasters:
        arr, file = read_raster_arr_object(raster)
        raster_list.append(file)

    # setting resampling method
    if resampling_method == 'nearest':
        resampling_method = Resampling.nearest
    elif resampling_method == 'bilinear':
        resampling_method = Resampling.bilinear

    # reference raster
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # merging
    if resolution is None:  # will use first input raster's resolution
        merged_arr, out_transform = merge(raster_list, bounds=ref_file.bounds,
                                          resampling=resampling_method, method=mosaicking_method,
                                          nodata=nodata)

    else:  # will use input resolution
        merged_arr, out_transform = merge(raster_list, bounds=ref_file.bounds, res=(resolution, resolution),
                                          resampling=resampling_method, method=mosaicking_method, nodata=nodata)

    # nodata operation
    # merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_raster = output_dir / raster_name
    write_array_to_raster(raster_arr=merged_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=out_raster, nodata=nodata, ref_file=ref_raster, dtype=dtype)

    return merged_arr, out_raster


def mosaic_raster_list(input_raster_list, output_dir, raster_name, ref_raster=WestUS_raster, dtype=None,
                       resampling_method='nearest', mosaicking_method='first', resolution=None,
                       nodata=no_data_value):
    """
    Mosaics a list of input rasters.


    :param input_raster_list: A list of input rasters to merge/mosaic.
    :param output_dir: Output raster directory.
    :param raster_name: Output raster name.
    :param ref_raster: Reference raster filepath. Set default to WestUS_raster.
    :param dtype: Output raster data type. Default set to None.
    :param resampling_method: Resampling method. Default set to 'nearest'. Can also take 'bilinear'. Currently, can use
                              only these two resampling methods.
    :param mosaicking_method: Mosaicking method. Can be 'first' or 'max' or 'min'. Default set to 'first'.
    :param resolution: Resolution of the output raster. Default set to None to use the first input raster's resolution.
    :param nodata: no_data_value set as -9999.

    :return: Mosaicked raster array and filepath of mosaicked raster.
    """
    raster_file_list = []  # a list to store raster file information
    for raster in input_raster_list:
        arr, file = read_raster_arr_object(raster)
        raster_file_list.append(file)

    # setting resampling method
    if resampling_method == 'nearest':
        resampling_method = Resampling.nearest
    elif resampling_method == 'bilinear':
        resampling_method = Resampling.bilinear

    # reference raster
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # merging
    if resolution is None:  # will use first input raster's resolution
        merged_arr, out_transform = merge(raster_file_list, bounds=ref_file.bounds,
                                          resampling=resampling_method, method=mosaicking_method,
                                          nodata=nodata)

    else:  # will use input resolution
        merged_arr, out_transform = merge(raster_file_list, bounds=ref_file.bounds, res=(resolution, resolution),
                                          resampling=resampling_method, method=mosaicking_method, nodata=nodata)

    # nodata operation
    merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    # saving output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_raster = output_dir / raster_name
    write_array_to_raster(raster_arr=merged_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=out_raster, nodata=nodata, ref_file=ref_raster, dtype=dtype)

    return merged_arr, out_raster


def clip_resample_reproject_raster(input_raster, input_shape, output_raster_dir,
                                   keyword=' ', raster_name=None,
                                   clip=False, resample=False, clip_and_resample=True,
                                   targetaligned=True, resample_algorithm='near',
                                   resolution=None,
                                   crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                   use_ref_width_height=False, ref_raster=WestUS_raster):
    """
    Clips, resamples, reprojects a given raster using input shapefile, resolution, and crs.

    ** If resolution is None, must provide a left_zone_ref_raster. One of resolution and left_zone_ref_raster must be available.

    :param input_raster: Input raster filepath.
    :param input_shape: Input shape filepath. Set to None when resample=True.
    :param keyword: 'str' keyword to attach in front of processed raster. Default set to ' '.
                    ** Only works when raster_name = None.
    :param raster_name: Output raster name. Default set to None to set raster name from the input raster.
    :param output_raster_dir: Output directory filepath.
    :param clip: Set to True to clip only. When True, resample and clip_and_resample should be False. resolution can be
                 used as None (Default) in this case.
    :param resample: Set to True to resample only. When True, clip and clip_and_resample should be False. Need to set
                     a resolution.
    :param clip_and_resample: Set to True to both clip and resample. When True, clip and resample should be False. Need
                              to set a resolution. Set basin_shape to None.
    :param targetaligned: Set to False if pixels don't need to be aligned in case of clip_and_resample=True. Look into
                          the result when using this to be sure about expected outcome.
    :param resample_algorithm: Resample algorithm to use in resampling. Can take near/bilinear/average/mode/max/min/cubic etc.
                               Default is 'near'.
    :param resolution: Output raster resolution. Default set to None (for clip=True). For other purposes, use specific
                       resolution value (for example, 0.02000000000000000389 deg).
    :param crs: Output raster projection. Default set to 'EPSG:4269' (NAD83).
    :param output_datatype: Output data type. Default set to gdal.GDT_Float32.
    :param use_ref_width_height: Set to True to use reference raster's width+height for resampling/clipping,
                                 instead of a particular assigned resolution. 'resolution' can be set to None.
    :param ref_raster: Filepath of reference raster to be used for assigning processed raster's width+height.

    :return: Processed raster filepath.
    """
    global processed_data

    if raster_name is None:  # if raster_name is None will set raster name from the input raster.
        raster_name = Path(input_raster).name
        output_raster_name = f"{keyword}_{raster_name}"
        if keyword == ' ':
            output_raster_name = raster_name
    else:  # to assign assigned raster_name. If None, will figure out raster_name using the first conditional if.
        if '.tif' not in raster_name:
            raster_name = raster_name + '.tif'
        output_raster_name = raster_name

    # creating output directory
    output_raster_dir = Path(output_raster_dir)
    output_raster_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_raster_dir / output_raster_name

    # opening input raster
    raster_file = gdal.Open(input_raster)

    if clip:  # set resample, clip_and_resample = False
        # resolution argument can be None in clip operation
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       cutlineDSName=input_shape,
                                       cropToCutline=True, dstNodata=no_data_value, outputType=output_datatype)
        else:
            _, xres, _, _, _, yres = raster_file.GetGeoTransform()
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=xres, yRes=yres,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       outputType=output_datatype)

    elif resample:  # set clip, clip_and_resample = False
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            # basin_shape can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       dstNodata=no_data_value, resampleAlg=resample_algorithm,
                                       outputType=output_datatype)
        else:
            # have to provide a resolution value in argument
            # basin_shape can be set to None
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                       dstNodata=no_data_value, resampleAlg=resample_algorithm,
                                       outputType=output_datatype)

    elif clip_and_resample:  # set clip=False, resample = False
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       resampleAlg=resample_algorithm, outputType=output_datatype)
        else:
            # argument must have basin_shape and resolution value
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       resampleAlg=resample_algorithm, outputType=output_datatype)
    del processed_data

    return output_filepath


def shapefile_to_raster(input_shape, output_dir, raster_name, burnvalue=None, use_attr=True,
                        attribute="", add=None, ref_raster=WestUS_raster,
                        resolution=model_res, alltouched=False):
    """
    Converts polygon shapefile to raster by attribute value or burn value.

    :param input_shape: Filepath of input shapefile.
    :param output_dir: Filepath of output directory.
    :param raster_name: Output raster name.
    :param burnvalue: Value for burning into raster. Only needed when use_attr is False. Default set to None.
    :param use_attr: Set to True if raster needs to be created using a specific attribute value. Defaults to False.
    :param attribute: Attribute name to use creating raster file. Defaults to "".
    :param add: Set to True if all values inside the raster grid should be summed. Default set to None to perform
                rasterizing with an attribute without summing.
    :param ref_raster: Reference raster to get minx, miny,maxx, maxy. Defaults to rWestUS_raster.
    :param resolution: Resolution of the output raster. Defaults to model_res of ~0.02.
    :param alltouched: If True all pixels touched by lines or polygons will be updated.

    :return: Filepath of created raster.
    """
    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    total_bounds = ref_file.bounds

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_raster = output_dir / raster_name

    if use_attr:
        if add is not None:  # adds point values in the pixel
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=no_data_value, attribute=attribute, allTouched=alltouched,
                                                   initValues=0, add=True)
            ds = gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape, options=raster_options,
                                resolution=resolution)

            # the rasterize sets 0 initialization at nodata pixels too
            # reapplying nodata values
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            arr[arr == 0] = no_data_value  # careful: this kills *real* zeros too
            band.WriteArray(arr)
            band.SetNoDataValue(no_data_value)
            band.FlushCache()

        else:  # the value of the last point's attribute (in case of overlapping or multiple pts in a pixel)
            # will be burned in the raster
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=no_data_value, attribute=attribute, allTouched=alltouched)
            gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape, options=raster_options,
                           resolution=resolution)

    else:  # will burn a specified value in the pixel
        raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                               outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                               noData=no_data_value, burnValues=burnvalue,
                                               allTouched=alltouched)
        gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape, options=raster_options,
                       resolution=resolution)

    return output_raster


def sum_rasters(raster_dir, output_raster, raster_list=None, search_by='*.tif', ref_raster=WestUS_raster,
                nodata=no_data_value):
    """
    Sum multiple rasters together. Can take raster directory or list of rasters as input.

    :param raster_dir: Filepath of input rasters' directory. When not using (using raster_list param) set to None.
    :param raster_list: A list of rasters to sum. Can alternatively be used with raster_dir. Default set to None. While
                        using set raster_dir=None.
    :param output_raster: Filepath of output raster.
    :param search_by: search by criteria to select raster from a directory.
    :param ref_raster: Reference raster filepath. Set default to WestUS_raster.
    :param nodata: no_data_value set as -9999.

    :return: Summed array and output raster.
    """
    if raster_dir is not None:
        input_rasters = list(Path(raster_dir).glob(search_by))
    else:
        input_rasters = raster_list

    sum_arr, file = None, None
    for raster in input_rasters:
        if raster == input_rasters[0]:
            arr = read_raster_arr_object(raster, get_file=False)
            sum_arr = arr
        else:
            arr = read_raster_arr_object(raster, get_file=False)
            sum_arr = np.nansum(np.dstack((sum_arr, arr)), axis=2)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    sum_arr[np.isnan(ref_arr)] = nodata  # setting nodata using reference raster

    write_array_to_raster(raster_arr=sum_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)

    return sum_arr, output_raster


def mean_rasters(raster_dir, output_raster, raster_list=None, search_by='*.tif', ref_raster=WestUS_raster,
                 nodata=no_data_value):
    """
    Calculate mean of multiple rasters. Can take raster directory or list of rasters as input.

    :param raster_dir: Filepath of input rasters' directory. When not using (using raster_list param) set to None.
    :param raster_list: A list of rasters to sum. Can alternatively be used with raster_dir. Default set to None. While
                        using set raster_dir=None.
    :param output_raster: Filepath of output raster.
    :param search_by: search by criteria to select raster from a directory.
    :param ref_raster: Reference raster filepath. Set default to WestUS_raster.
    :param nodata: no_data_value set as -9999.

    :return: Mean raster.
    """
    if raster_dir is not None:
        input_rasters = list(Path(raster_dir).glob(search_by))
    else:
        input_rasters = raster_list

    sum_arr, file = None, None
    val = 0
    for raster in input_rasters:
        if raster == input_rasters[0]:
            arr = read_raster_arr_object(raster, get_file=False)
            sum_arr = arr
            val += 1
        else:
            arr = read_raster_arr_object(raster, get_file=False)
            sum_arr += arr
            val += 1

    mean_arr = sum_arr / val
    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    mean_arr[np.isnan(ref_arr)] = nodata  # setting nodata using reference raster

    write_array_to_raster(raster_arr=mean_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)


def filter_raster_on_threshold(input_raster, output_raster, threshold_value1, threshold_value2=None, assign_value=None,
                               nodata=no_data_value, refraster=WestUS_raster):
    """
    Filters raster based on threshold (and assigns value if given).

    :param input_raster: Filepath of input raster.
    :param output_raster: Filepath of filtered/modified output raster.
    :param threshold_value1: Threshold value no. 1. Can be integer/float.
    :param threshold_value2: Threshold value no. 2. Can be integer/float. Default set to None so that only
                             threshold_value1 is used.
    :param assign_value: Assign a value to the filtered raster pixels. Default set no None to keep the original filtered
                         values.
    :param nodata: no_data_value set as -9999.
    :param refraster: Reference raster filepath. Set default to WestUS_raster.

    :return: Output raster filepath.
    """
    ref_arr, ref_file = read_raster_arr_object(refraster)
    input_arr = read_raster_arr_object(input_raster, get_file=False)

    mod_arr = None  # new array where the filtered array will be stored
    if assign_value is None:
        mod_arr = np.where(input_arr >= threshold_value1, input_arr, 0)
        mod_arr[np.isnan(ref_arr)] = nodata
    elif threshold_value2 is not None:
        mod_arr = np.where((input_arr >= threshold_value2) & (input_arr <= threshold_value1), input_arr, 0)
        mod_arr[np.isnan(ref_arr)] = nodata
    elif assign_value is not None:
        mod_arr = np.where(input_arr >= threshold_value1, assign_value, 0)
        mod_arr[np.isnan(ref_arr)] = nodata
    elif (threshold_value2 is not None) & (assign_value is not None):
        mod_arr = np.where((input_arr >= threshold_value2) & (input_arr <= threshold_value1), assign_value, 0)
        mod_arr[np.isnan(ref_arr)] = nodata

    write_array_to_raster(raster_arr=mod_arr, raster_file=ref_file, transform=ref_file.transform,
                          output_path=output_raster)
    return output_raster


def make_lat_lon_array_from_raster(input_raster, nodata=-9999):
    """
    Make lat, lon array for each pixel using the input raster.

    params:
    input_raster : Input raster filepath that will be used as reference raster.
    nodata : No data value. Default set to -9999.

    returns: Lat, lon array with nan value (-9999) applied.
    """
    raster_file = rio.open(input_raster)
    raster_arr = raster_file.read(1)

    # calculating lat, lon of each cell centroid
    height, width = raster_arr.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rio.transform.xy(rows=rows, cols=cols, transform=raster_file.transform)

    # flattening and reshaping to the input_raster's array size
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()

    lon_arr = xs.reshape(raster_arr.shape)
    lat_arr = ys.reshape(raster_arr.shape)

    # assigning no_data_value
    lon_arr[raster_arr == nodata] = nodata
    lat_arr[raster_arr == nodata] = nodata

    return lon_arr, lat_arr


def create_ref_raster(input_raster, output_ref_raster):
    """
    create reference raster (all valid value locations are zero) from an input raster.

    :param input_raster: Filepath of input raster.
    :param output_ref_raster: Filepath of output reference raster.

    :return: None.
    """
    input_arr, input_file = read_raster_arr_object(input_raster)

    # assign 0 where the value isn't nan (-9999) in the input raster
    input_arr = np.where(np.isnan(input_arr), -9999, 0)
    write_array_to_raster(raster_arr=input_arr, raster_file=input_file, transform=input_file.transform,
                          output_path=output_ref_raster)


def compute_proximity(input_raster, output_raster, target_values=(1,), nodatavalue=-9999):
    """
    Computes proximity (Euclidean distance) raster from target pixel values in the input raster.

    :param input_raster : Path to the input raster file.
    :param output_raster : Output proximity raster filepath.
    :param target_values : Tuple of target values to compute proximity from. Defaults to (1,).
    :param nodatavalue : No-data value to set in the output raster. Defaults to -9999.

    Returns: Path to the output proximity raster.
    """
    inras_file = gdal.Open(input_raster, gdal.GA_ReadOnly)
    inras_band = inras_file.GetRasterBand(1)
    driver = gdal.GetDriverByName('GTiff')

    x_size = inras_file.RasterXSize
    y_size = inras_file.RasterYSize
    dest_ds = driver.Create(output_raster, x_size, y_size, 1, gdal.GDT_Float32)
    dest_ds.SetProjection(inras_file.GetProjection())
    dest_ds.SetGeoTransform(inras_file.GetGeoTransform())
    dest_band = dest_ds.GetRasterBand(1)
    dest_band.SetNoDataValue(nodatavalue)

    target_values_list = list(target_values)
    values = 'VALUES=' + ','.join(str(val) for val in target_values_list)

    gdal.ComputeProximity(inras_band, dest_band, [values, "DISTUNITS=GEO"])

    inras_file, inras_band, dest_ds, dest_band = None, None, None, None

    return output_raster


def rasterize_shape_to_match(input_shape, ref_raster, burn_value=1, fill_value=0):
    """
    Rasterizes a shapefile to match a reference raster.

    Parameters
    ----------
    input_shape : str
        Path to shapefile (e.g., Basin & Range extent).
    ref_raster : str
        Path to reference raster file to match grid (same resolution, extent, CRS).
    burn_value : int, optional
        Value assigned to pixels inside the shape (default=1).
    fill_value : int, optional
        Value assigned to pixels outside the shape (default=0).

    Returns
    -------
    numpy.ndarray
        2D array mask aligned with ref_raster.
    """

    # open shapefile and reference raster
    gdf = gpd.read_file(input_shape)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    # reproject shapefile if needed
    if gdf.crs != ref_file.crs:
        gdf = gdf.to_crs(ref_file.crs)

    # rasterize
    mask_arr = features.rasterize(
                [(geom, burn_value) for geom in gdf.geometry],
                out_shape=ref_arr.shape,
                transform=ref_file.transform,
                fill=fill_value,
                dtype=np.uint8
            )

    return mask_arr