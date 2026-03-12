# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
import logging
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from pathlib import Path
from rasterstats import zonal_stats

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


def create_monthly_panel_data(
        years_list,
        aquifer_state_shapefile,
        aquifer_state_name_col,
        aquifer_region_col,
        aquifer_name_col,
        state_name_col,
        irrigated_cropland_dir,
        monthly_data_dirs,
        annual_data_dirs,
        static_data_dirs,
        output_csv_path,
        column_rename=None,
        include_zero_cols=None,
        growing_season_months=range(4, 11),
        no_data_value=-9999,
        skip_processing=False):
    """
    Aggregate raster datasets to aquifer-state unit level using exact polygon
    boundaries from a shapefile (via rasterstats) and construct a monthly panel
    DataFrame. Each row represents one (unit × year × month) observation.

    All variables are aggregated over IRRIGATED PIXELS ONLY within each polygon,
    using the annual irrigated cropland classification as a pixel-level pre-mask
    (irr == 1 strictly) before running zonal statistics.

    Zero and nodata handling
    ------------------------
    - Nodata (-9999) is ALWAYS excluded for all variables. Converted to NaN on load.
    - Zero values are excluded by default for all variables.
      Exception: columns listed in `include_zero_cols` retain zero-valued pixels.
      Use for precipitation, where zero is physically meaningful (no rainfall).
    - Example: include_zero_cols=['Precip_mm']

    IMPORTANT: All rasters must be co-registered to the same grid (same CRS,
    resolution, and extent) as the irrigated cropland raster. The irrigated mask
    is applied element-wise before zonal aggregation. A shape mismatch will raise
    a ValueError.

    Variable config format
    ----------------------
    Each variable is a 2-element tuple:
        (directory_or_path, aggregation_method)

    Aggregation methods:
        'mean'   — mean   over valid pixels
        'median' — median over valid pixels
        'sum'    — sum    over valid pixels

    Example inputs
    --------------
        aquifer_state_shapefile = PROJECT_ROOT / 'Data_main/shapefiles/aquifer_state_units.shp'
        aquifer_state_name_col  = 'AQ_State'
        aquifer_name_col        = 'AQ_code'
        aquifer_region_col      = 'AQ_Name'
        state_name_col          = 'State'

        monthly_data_dirs = {
            'ET_mm'     : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly',              'mean'),
            'IWU_v1_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v1_current',       'mean'),
            'IWU_v2_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v2_current_prev1', 'mean'),
            'IWU_v3_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v3_current_prev2', 'mean'),
            'Precip_mm' : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked',           'mean'),
            'Tmean_C'   : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Tmean/monthly',                   'mean'),
        }

        annual_data_dirs = {
            'Irr_area_ha' : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_area', 'sum'),
        }

        static_data_dirs = {
            'WTD_Rnd_Frst_m' : (PROJECT_ROOT / 'Data_main/rasters/CONUS_WTD_RF',          'median'),
            'WTD_USGS_m'     : (PROJECT_ROOT / 'Data_main/rasters/USGS_Unconfined_WTD',   'median'),
        }

        include_zero_cols = ['Precip_mm']

    :param years_list: List of years to process.
    :param aquifer_state_shapefile: Path to aquifer-state polygon shapefile.
                                     Each row = one aquifer-state unit.
    :param aquifer_state_name_col: Shapefile column for aquifer-state unit name
                                    (e.g. 'AQ_State' → 'BR_AZ', 'CP_OR').
    :param aquifer_region_col: Shapefile column for aquifer region/name
                                (e.g. 'AQ_Region' → 'CV_CA_Sacramento', 'HPA_KS_East').
    :param aquifer_name_col: Shapefile column for aquifer code
                              (e.g. 'AQ_code' → 'BR', 'CP', 'HPA').
    :param state_name_col: Shapefile column for state name (e.g. 'State' → 'Arizona').
    :param irrigated_cropland_dir: Directory of annual irrigated cropland rasters.
                                    Pattern: *{year}*.tif  (1=irrigated, -9999=nodata).
                                    Only pixels with value == 1 are treated as irrigated.
    :param monthly_data_dirs: Dict of {col: (directory, agg_method)} for monthly rasters.
    :param annual_data_dirs: Dict of {col: (directory, agg_method)} for annual rasters.
    :param static_data_dirs: Dict of {col: (directory, agg_method)} for static rasters.
    :param output_csv_path: Path to save the output panel CSV.
    :param column_rename: Optional dict to rename output DataFrame columns.
    :param include_zero_cols: List of column names where zero values should be retained
                               during aggregation (e.g. ['Precip_mm']). All other columns
                               have zeros excluded by default.
    :param growing_season_months: Months to process. Default: April–October (range(4, 11)).
    :param no_data_value: Nodata value used across all rasters. Default: -9999.
    :param skip_processing: If True, skip this step and return None.

    :return: pd.DataFrame of the monthly panel, or None if skipped.
    """
    if skip_processing:
        return None

    # 
    include_zero_cols = set(include_zero_cols) if include_zero_cols else set()

    # -------------------------------------------------------------------------
    # helper: normalise config tuple to (Path, agg_method)
    # -------------------------------------------------------------------------
    def parse_config(config_dict):
        if config_dict is None: # in case an empty config is passed, avoid iterating over None 
            return {}
        
        return {k: (Path(v[0]), v[1]) for k, v in config_dict.items()}

    # -------------------------------------------------------------------------
    # helper: load raster as float32 array + affine transform
    #         nodata → NaN on load so rasterstats sees NaN as the nodata value
    # -------------------------------------------------------------------------
    def load_arr(path):
        """Returns (array, transform) or (None, None) if file missing."""
        if path is None or not Path(path).exists():
            return None, None
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            transform = src.transform
        arr[arr == no_data_value] = np.nan
        return arr, transform

    # -------------------------------------------------------------------------
    # helper: find first file matching glob pattern
    # -------------------------------------------------------------------------
    def find_file(directory, pattern):
        matches = list(directory.glob(pattern))
        return matches[0] if matches else None

    # -------------------------------------------------------------------------
    # helper: apply irrigated mask + zero exclusion to array
    #         result is passed directly to rasterstats (NaN = nodata)
    # -------------------------------------------------------------------------
    def apply_irr_mask(arr, irr_mask, col_name):
        """
        Set non-irrigated pixels to NaN (irr_mask==False).
        Also set zeros to NaN unless col_name is in include_zero_cols.
        Array and irr_mask must have identical shapes (same grid required).
        """
        if arr.shape != irr_mask.shape:
            raise ValueError(
                f'Shape mismatch for "{col_name}": array={arr.shape}, '
                f'irr_mask={irr_mask.shape}. All rasters must share the same grid.'
            )
    
        # setting non-irrigated pixels to NaN ensures rasterstats 'count' = irrigated pixel count
        out = arr.copy()
        out[~irr_mask] = np.nan                       # non-irrigated → NaN
        
        # set zero values to NaN, except for precipitation
        if col_name not in include_zero_cols:
            out[out == 0] = np.nan                    # zeros → NaN (fallow / no demand)

        return out

    # -------------------------------------------------------------------------
    # helper: run zonal_stats on a pre-masked array, return list of values
    #         (one value per polygon, same order as gdf rows)
    # -------------------------------------------------------------------------
    def run_zonal(arr, transform, stat):
        results = zonal_stats(
            gdf.geometry, arr, affine=transform,
            nodata=np.nan, stats=[stat]
        )
        # rasterstats returns None (not NaN) when no valid pixels exist
        return [r.get(stat) if r.get(stat) is not None else np.nan
                for r in results]

    logger.info('---------------------------------------------------------------')
    logger.info(f'Starting to compile monthly panel dataframe')
 
    
    # -------------------------------------------------------------------------
    # path setup
    # -------------------------------------------------------------------------
    irrigated_cropland_dir = Path(irrigated_cropland_dir)
    output_csv_path        = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    monthly_data_path_dict = parse_config(monthly_data_dirs)
    annual_data_path_dict  = parse_config(annual_data_dirs)
    static_data_path_dict = parse_config(static_data_dirs)

    # -------------------------------------------------------------------------
    # load shapefile 
    # -------------------------------------------------------------------------
    gdf = gpd.read_file(aquifer_state_shapefile)

    # ------------------------------------------------------------------------
    # main processing loop
    # -------------------------------------------------------------------------
    
    # empty dictionary for storing extraced data from annual/static/monthly datasets
    # need to handle the case where annual_data_path_dict returns an empty dict
    all_panel_cols = list(annual_data_path_dict.keys()) + list(monthly_data_path_dict.keys()) + \
        list(static_data_path_dict.keys()) + ['aquifer_state', 'aquifer_region', 'aquifer', 'state', 'year', 'month']
    results_dict = {col: [] for col in all_panel_cols}

    for year in years_list:
        logger.info(f'Processing year={year}...')

        # load irrigated cropland classification
        # irr == 1 → irrigated; 0 or NaN → excluded
        irr_file = find_file(irrigated_cropland_dir, f'*{year}*.tif')
        irr_arr, irr_transform = load_arr(irr_file)
        irr_mask = (irr_arr == 1)   # strictly irrigated pixels only

        #-------------------------------------------------------------
        # extract annual data for this year
        #-------------------------------------------------------------
        
        if annual_data_dirs is None: # in case of empty config for annual data
            logger.info('No annual data directories provided — skipping annual variables.')
            pass
        
        else:            
            for col, (path_or_dir, agg) in annual_data_path_dict.items():
                fpath = find_file(path_or_dir, f'*{year}*.tif')
 
                if fpath is None:
                    logger.warning(f'Annual data missing: col="{col}", year={year}.')
                    results_dict[col].extend([np.nan] * len(gdf) * len(list(growing_season_months)))
                    continue
                
                arr, transform = load_arr(fpath)
                masked = apply_irr_mask(arr, irr_mask, col)
                vals = run_zonal(masked, transform, agg)
                results_dict[col].extend(vals * len(growing_season_months))

        #-------------------------------------------------------------
        # extract static data for this year
        #-------------------------------------------------------------
        for col, (path_or_dir, agg) in static_data_path_dict.items():
            fpath = find_file(path_or_dir, f'*.tif')

            if fpath is None:
                logger.warning(f'Static data missing: col="{col}"')
                results_dict[col].extend([np.nan] * len(gdf) * len(growing_season_months))
                continue
            
            arr, transform = load_arr(fpath)
            masked = apply_irr_mask(arr, irr_mask, col)
            vals   = run_zonal(masked, transform, agg) # here we are not masking with irr_mask; some WTD data is CONUS wide
            results_dict[col].extend(vals * len(growing_season_months))

        # ----------------------------------------------------------------------
        # extract monthly data for this year
        # ----------------------------------------------------------------------
        for month in growing_season_months:

            # load monthly variable arrays
            for col, (d, agg) in monthly_data_path_dict.items():
                fpath = find_file(d, f'*{year}_{month}.tif')
                
                if fpath is None:
                    logger.warning(f'Monthly data missing: col="{col}", year={year}, month={month}.')
                    results_dict[col].extend([np.nan] * len(gdf)) # add NaN for all units for this month
                    continue
                
                arr, transform = load_arr(fpath)
                masked = apply_irr_mask(arr, irr_mask, col)
                vals = run_zonal(masked, transform, agg)
                results_dict[col].extend(vals)

            # ----------------------------------------------------------------------
            # add aquifer-state/aquifer/state/year/month info for this month
            # ----------------------------------------------------------------------
            results_dict['aquifer_state'].extend(gdf[aquifer_state_name_col].values)
            results_dict['aquifer_region'].extend(gdf[aquifer_region_col].values)
            results_dict['aquifer'].extend(gdf[aquifer_name_col].values)
            results_dict['state'].extend(gdf[state_name_col].values)
            results_dict['year'].extend([year] * len(gdf))
            results_dict['month'].extend([month] * len(gdf))
            
    # -------------------------------------------------------------------------
    # build DataFrame
    # -------------------------------------------------------------------------
    panel_df = pd.DataFrame(results_dict)

    if column_rename:
        missing = [k for k in column_rename if k not in panel_df.columns]
        if missing:
            logger.warning(f'column_rename keys not found in DataFrame: {missing}')
            
        panel_df = panel_df.rename(columns=column_rename)

    panel_df.to_csv(output_csv_path, index=False)
    logger.info(f'Panel data saved → {output_csv_path}  |  shape: {panel_df.shape}')
    logger.info('---------------------------------------------------------------')

    return panel_df
