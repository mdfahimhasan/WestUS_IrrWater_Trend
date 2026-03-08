import os
import sys
import numpy as np
import geopandas as gpd
from pathlib import Path
from osgeo import gdal, osr, ogr
from shapely.geometry import Polygon

from Codes.utils.system_ops import makedirs
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


WestUS_raster = PROJECT_ROOT / 'Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def create_buffer(input_shapefile, distance, output_shapefile, change_crs='EPSG:32611'):
    """
    creates a buffer region around a shapefile.

    :param input_shapefile: Filepath of input shapefile.
    :param distance: Value of distance. Unit will be in the unit of the crs.
    :param output_shapefile: Filepath of output (buffered) shapefile.
    :param change_crs: Default set to 'EPSG:32611' (UTM zone 11N) as the projected crs.

    :return: Returns the filepath of buffered shapefile.
    """
    input_gdf = gpd.read_file(input_shapefile)
    original_crs = input_gdf.crs
    if change_crs is not None:
        input_gdf = input_gdf.to_crs(change_crs)

    input_gdf = input_gdf.buffer(distance)
    input_gdf = input_gdf.to_crs(original_crs)  # converting back to original crs
    input_gdf.to_file(output_shapefile)

    return output_shapefile


def clip_vector(input_shapefile, mask_shapefile, output_shapefile, change_crs=None, create_zero_buffer=False):
    """
    Clips a vector file based on the vector mask provided.

    :param input_shapefile: Filepath of input shapefile.
    :param mask_shapefile: Filepath of shapefile to use as mask.
    :param change_crs: Crs if want to change crs. Currently changes input shapefile's crs.
    :param output_shapefile: Filepath of output (clipped) shapefile.
    :param create_zero_buffer: Set to True to create zero buffer around input shapefile. Default set to False.
                               There might be intersection error for complex/problematic shapefiles. Creating a zero
                               buffer around it solves the issue. If it doesn't further investigate.

    :return: Returns the filepath of clipped shapefile.
    """
    output_dir = os.path.dirname(output_shapefile)
    makedirs([output_dir])

    if create_zero_buffer:
        # # Buffering might ruin the attribute information of the input shapefile. Do further processing.
        buffered_shape_path = os.path.join(output_dir, 'zero_buffer.shp')

        # Creating zero buffered shapefile. The crs will be converted to original crs of the shapefile automatically
        buffered_shapefile = create_buffer(input_shapefile, distance=0, output_shapefile=buffered_shape_path,
                                           change_crs=change_crs)
        input_shapefile = buffered_shapefile

    input_gdf = gpd.read_file(input_shapefile)
    mask_gdf = gpd.read_file(mask_shapefile)

    if change_crs is not None:
        input_gdf = input_gdf.to_crs(change_crs)

    clipped_gdf = gpd.clip(input_gdf, mask_gdf, keep_geom_type=True)

    clipped_gdf.to_file(output_shapefile)

    return output_shapefile


def create_pixel_multipoly_shapefile(refraster, interim_output_raster, output_file):
    """
    Creates a shapefile of polygon each having a width of the input raster's pixel size.
    ** This code will generate the polygon for the whole extent of the input raster file. Clip it with appropriate
    shapefile in python/gis to get the intended polygon shapefile.

    :param refraster: Filepath of reference raster.
    :param interim_output_raster: Filepath of intermediate output raster where each pixel has unique DNv alue.
    :param output_file: Filepath of output polygon.

    :return: A multi-polygon with each polygon having a width of the input raster's pixel size.
    """

    # getting total polygons estimate
    ref_arr, ref_file = read_raster_arr_object(refraster)
    shape = ref_arr.shape
    total_pol = (shape[0] * shape[1]) + 1  # number of total polygons to create based on no. members in ref raster

    # the new_arr will have individual pixels with unique DN values
    new_arr = np.arange(start=1, stop=total_pol, step=1).reshape(shape)

    # creating scratch dir in case the folder doesn't exist
    makedirs(['../../scratch'])
    write_array_to_raster(new_arr, ref_file, ref_file.transform, interim_output_raster)

    # converting the newly created raster to polygon
    raster = gdal.Open(interim_output_raster)
    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()

    proj = raster.GetProjection()
    shp_proj = osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)

    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    create_shp = call_drive.CreateDataSource(output_file)
    shp_layer = create_shp.CreateLayer('layername', srs=shp_proj)
    new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
    shp_layer.CreateField(new_field)

    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
    create_shp.Destroy()

    raster = None


def create_fishnets_from_shapefile(input_shape, num_cols, num_rows, output_shape, crs=None):
    """
    Create fishnet polygons' shapefile using the extent of input shapefile.

    :param input_shape: Filepath of input shapefile from which extent will be extracted.
    :param num_cols: Number of columns in the fishnet.
    :param num_rows: Number of rowss in the fishnet.
    :param output_shape: Filepath of output fishnet shapefile.
    :param crs: CRS in 'EPSG:4269' format. Default set to None to use crs from input shapefile.

    :return: None.
    """
    input_shp_gdf = gpd.read_file(input_shape)
    xmin, ymin, xmax, ymax = input_shp_gdf.total_bounds

    grid_x_size = abs((xmin - xmax) / num_cols)
    grid_y_size = abs((ymin - ymax) / num_rows)

    lats = list(np.arange(xmin, (xmax + grid_x_size), grid_x_size))
    lons = list(np.arange(ymin, (ymax + grid_y_size), grid_y_size))

    poly_geoms = []
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            bounds = Polygon(
                [(lats[i], lons[j]), (lats[i], lons[j + 1]), (lats[i + 1], lons[j + 1]), (lats[i + 1], lons[j]),
                 (lats[i], lons[j])])
            poly_geoms.append(bounds)

    if crs is None:
        crs = input_shp_gdf.crs

    fishnet = gpd.GeoDataFrame(poly_geoms, columns=['geometry']).set_crs(crs)
    fishnet.to_file(output_shape)
