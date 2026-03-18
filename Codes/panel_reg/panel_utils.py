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
import pyfixest as pf
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


def create_monthly_panel_dataframe(
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
            'WTD_Rnd_Frst_m'   : (PROJECT_ROOT / 'Data_main/rasters/CONUS_WTD_RF',          'median'),
            'WTD_USGS_m'       : (PROJECT_ROOT / 'Data_main/rasters/USGS_Unconfined_WTD',   'median'),
            'GW_or_conjunctive' : (PROJECT_ROOT / 'Data_main/rasters/GW_use_%/GW_use_binary/GW_use_perc_ROI_final.tif', 'median')
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
        if '.tif' in directory.name:  # if a file path is given instead of a directory
            return directory
        
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
    logger.info(f'Raw Panel dataframe saved → {output_csv_path}  |  shape: {panel_df.shape}')
    logger.info(('*** Further processing might be required before regression model ***'))
    logger.info('---------------------------------------------------------------')

    return panel_df


def compute_anomaly_in_df(df, regressors_for_anomalies_dict,
                          unit_col='aquifer_region', 
                          year_col='year', month_col='month'):
    """
    Calculate anomalies for a monthly panel dataframe used in pyfixest regression.

    This function computes climate baselines and anomalies for specified regressors.

    *** This function does not create FE columns for pyfixest FE regreesion.
        Must run create_FE_columns() separately after this step. ***
    
    Example inputs:
    ---------------
        df = pd.read_csv('path/to/panel_df.csv')
        
        regressors_for_anomalies_dict = {'Precip_mm': [1986, 2000], 'Tmean_C': [1986, 2000]}
        
        
    :param df: Monthly panel dataframe containing climate and IWU variables.
    
    :param regressors_for_anomalies_dict: Dictionary mapping column names to
        baseline periods. Keys are column names to compute anomalies for,
        values are a list or range of two integers [start_year, end_year].
        Example: {'Precip_mm': [1986, 2000], 'Tmean_C': [1986, 2000]}
        
    :param unit_col: Column name for spatial unit identifier. Default: 'aquifer_region'.
    :param year_col: Column name for year. Default: 'year'.
    :param month_col: Column name for month. Default: 'month'.
    
    :return: Input dataframe with additional columns:
        - '{prefix}_baseline': unit-month mean over the baseline period
        - '{prefix}_anomaly' : observed value minus baseline mean
        
        Returns None if any baseline period is invalid — check logger for details.
    """
    
    # nodata/NaN removal
    df = df.dropna()
    
    valid = True
    
    # calculate climate baselines for anomaly estimation
    for col, periods in regressors_for_anomalies_dict.items():
        
        # checking if periods is a list of two integers (start_year, end_year)
        periods = list(periods)

        if len(periods) != 2:
            logger.error(f'Invalid baseline periods for {col}: {periods}. ' 
                         f'Dict values must be a list or range of two integers: [start_year, end_year].')
            valid = False
            break
    
        column_prefix = col.split('_')[0]  # e.g. 'ET' from 'ET_mm'
        
        # calculate baseline mean for this variable and add as new column
        baseline = (df[df[year_col].between(periods[0], periods[1])]
                    .groupby([unit_col, month_col])[col]
                    .mean()).rename(column_prefix + '_baseline').reset_index()
        
        # merge baseline back to main df
        df = df.merge(baseline, on=[unit_col, month_col], how='left')
        
    # calculate anomalies by subtracting baseline from original value
    if not valid:
        return None

    for col in regressors_for_anomalies_dict.keys():
        column_prefix = col.split('_')[0]
        baseline_col = column_prefix + '_baseline'
        anomaly_col = column_prefix + '_anomaly'
        df[anomaly_col] = df[col] - df[baseline_col]
        
    logger.info(f'STEP 1: Anomaly columns created for: {list(regressors_for_anomalies_dict.keys())}\n')

    return df


def mean_WTD_col_for_unit(df, WTD_col, unit_col='aquifer_region'):
    
    """
    Replace annual WTD values at the unit-level with the mean across that unit for all years.
    
    Aggregating to unit-level mean avoids endogeneity from year-to-year
    WTD changes driven by pumping history, and reflects the near-zero
    within-unit annual WTD variation relative to cross-unit gradient.

    :param df: Monthly panel dataframe.
    :param WTD_col: Column name of the WTD variable to aggregate (e.g. 'WTD_Rnd_Frst_m').
    :param unit_col: Column name for spatial unit identifier. Default: 'aquifer_region'.

    :return: Dataframe with WTD_col replaced by 'WTD_mean_m' — unit-level mean WTD.
    """

    if WTD_col not in df.columns:
        raise ValueError(f'WTD column "{WTD_col}" not found in dataframe.')

    # compute long-term mean WTD per unit and rename before merging
    group_df = (df.groupby(unit_col)[WTD_col]
                .mean()
                .reset_index()
                .rename(columns={WTD_col: 'WTD_mean_m'}))  # ← rename here

    df = df.drop(columns=[WTD_col])
    df = df.merge(group_df, on=unit_col, how='left')

    logger.info(f'WTD column "{WTD_col}" replaced with unit-level mean → "WTD_mean_m"\n')

    return df


def create_FE_columns_in_df(df, fe_config,
                            year_col='year', 
                            month_col='month'):
    """
    Create composite fixed effects columns and time_id for pyfixest panel regression.

    :param df: Monthly panel dataframe.
    :param fe_config: Dictionary mapping new FE column names to lists of columns
        to concatenate. Example:
        {
            'aquifer_region_month' : ['aquifer_region', 'month'],
            'aquifer_type_year'    : ['aquifer_type',   'year'],
        }
    :param year_col:  Column name for year.  Default: 'year'
    :param month_col: Column name for month. Default: 'month'

    :return: Dataframe with new FE columns and time_id column added.
    """

    for fe_col, source_cols in fe_config.items():
        df[fe_col] = df[source_cols[0]].astype(str)
        
        for col in source_cols[1:]:
            df[fe_col] = df[fe_col] + '_' + df[col].astype(str)

    # time_id always created — needed for Newey-West SE ordering in pyfixest
    df['time_id'] = (df[year_col].astype(str) + '-' +
                     df[month_col].astype(str).str.zfill(2))
    
    logger.info(f'STEP 2: Fixed effects columns created for: {list(fe_config.keys())}\n')

    return df


def create_categorical_cols_in_df(df, categorical_config):
    """
    Create pd.Categorical columns for use as interaction variables in pyfixest regression.

    Must be run before pyfixest_fit_FE() when using categorical interaction terms (RQ1, RQ4).
    Not required for binary integer interaction columns (RQ3).

    Example inputs:
    ---------------
        # RQ1 — nominal aquifer type, no reference category (include_base_regressors=False)
        categorical_config = {
            'aq_type_cat': {
                'col_name' : 'aquifer',
                'assigned_categories' : ['BR', 'CP', 'CV', 'DBA', 'HPA', 'RG', 'SRP'],
                'impose_order'    : False
            }
        }

        # RQ4 — ordered WTD class, shallow is reference (include_base_regressors=True)
        categorical_config = {
            'WTD_cat': {
                'col_name' : 'WTD_class',
                'assigned_categories' : ['shallow', 'moderate', 'deep'],
                'impose_order'    : True
            }
        }

    :param df: Monthly panel dataframe.
    :param categorical_config: Dictionary mapping new column names to config dicts.
        Each config dict must have:
            - 'col_name'  : str — source column to convert
            - 'assigned_categories'  : list — category levels in desired order.
                              First level = reference category when include_base_regressors=True.
            - 'impose_order'     : bool — True if categories have a natural ordinal relationship.

    :return: Dataframe with new pd.Categorical columns added.
        Returns ValueE if source column is missing or categories are invalid.
    """

    for new_col, config in categorical_config.items():

        col_name = config['col_name']
        assigned_categories = config['assigned_categories']
        impose_order    = config['impose_order']

        # validate source column exists
        if col_name not in df.columns:
            raise ValueError(f'Source column "{col_name}" not found in dataframe.')

        # validate all categories exist in the data
        available_categories_in_df = set(df[col_name].dropna().unique())
        missing_categoris = set(assigned_categories) - available_categories_in_df
        
        if missing_categoris:
            raise ValueError(f'Categories {missing_categoris} not found in dataframe column "{col_name}". '
                             f'Actual values: {available_categories_in_df}')

        df[new_col] = pd.Categorical(df[col_name], categories=assigned_categories,ordered=impose_order)

    logger.info(f'STEP 3: Categorical columns created: {list(categorical_config.keys())}\n')

    return df

def pyfixest_fit_FE(df, target_col, regressor_cols, fe_cols,
                    include_base_regressors=True, 
                    interaction_dict=None,
                    unit_col=None, trend_col=None,
                    vcov_col='time_id', vcov_method="driscoll-kraay", 
                    bandwidth=24):
    """
    Fit a pyfixest panel regression model with specified target, regressors,
    fixed effects, optional unit-specific linear trend, and robust SEs.

    :param df: DataFrame containing the panel data.
    
    :param target_col: Name of the target variable column.
    
    :param regressor_cols: List of regressor column names.
    
    :param fe_cols: List of fixed effects column names.
    
    :param include_base_regressors: If True, include regressor_cols as main effects in the formula.
    
    :param interaction_dict: Optional dictionary to assign interaction term. 
        The key is the regressor column name and the value is the column name to interact with.
        
        Example: {'ET_mm_anomaly': 'aquifer_region'}
        
        * If the interaction column is float or string type, a ValueError is raised.
          Convert to int or categorical codes before passing.
    
    :param unit_col: Column name for spatial unit (e.g. 'aquifer_region').
        Required if trend_col is specified.
        
    :param trend_col: Column name for linear trend variable (e.g. 'year').
        If provided with unit_col, adds a unit-specific linear trend as
        unit_col[trend_col] in the formula.
    
    :param vcov_col: Column name for SE estimation. Usage depends on vcov_method:
        - 'driscoll-kraay' : time-ordered column (e.g. 'time_id')
        - 'NW'             : time-ordered column (e.g. 'time_id')
        - 'CRV1'           : cluster column (e.g. 'aquifer_region')
        
        Default is 'time_id' for Driscoll-Kraay.
        
    :param vcov_method: SE estimation method. Default: 'driscoll-kraay'.
        - 'driscoll-kraay' : Robust to serial correlation, spatial correlation
                             across units, and heteroskedasticity. Uses a
                             kernel-based estimator with no minimum cluster
                             requirement. Recommended for small N panels with
                             shared climate shocks (ENSO, PDO). Default.
                             
                             Uses time_id as the organizing dimension for both
                             corrections: within each time period, tracks
                             whether residuals move together across units
                             (spatial correlation); across time periods,
                             tracks whether residuals are autocorrelated
                             (serial correlation). Bandwidth controls how
                             many time lags to account for.
                             
        - 'NW'             : Newey-West. Robust to serial correlation and
                             heteroskedasticity only. Treats units as
                             independent — does not account for spatial
                             correlation across units.
        - 'CRV1'           : Clustered SEs. Allows free correlation within
                             each cluster across all time periods. More flexible
                             than NW within units but ignores cross-unit
                             correlation. Asymptotic — requires large N to be
                             reliable (≥20-30 clusters).
    
    :param bandwidth: Number of time lags for autocorrelation correction.
        Only applies to 'driscoll-kraay' and 'NW'. Default: 24 (covers
        2 years of monthly autocorrelation).

    :return: Fitted pyfixest model object.
    """
    
    # base regressors
    # the else block handles the case (include_base_regressors=False) where interaction_dict is provided 
    # for the regressors and no separate base regressors are required
    regressors = ' + '.join(regressor_cols) if include_base_regressors else ''
        
    # add unit-specific linear trend if specified
    if unit_col and trend_col:
        trend_term = f'{unit_col}[{trend_col}]'
        regressors = f'{regressors} + {trend_term}' if regressors else trend_term
        
    # looping through regression columns and their interaction pairs in the interaction_dict
    # to validate interaction columns and build interaction terms for the formula syntax
    if interaction_dict:
        
        if not isinstance(interaction_dict, dict):
            raise ValueError(f'Invalid interaction_dict: {interaction_dict}. Must be a dictionary of regressor_col: interact_col pairs.')

        for reg_col, interact_col in interaction_dict.items():
            
            # the following if-elif block checks if the interaction column is numeric (int or float) or string type.
            # If float or string type, a ValueError is raised — convert to int or categorical codes before passing.

            if pd.api.types.is_categorical_dtype(df[interact_col]):
                pass
            
            elif pd.api.types.is_integer_dtype(df[interact_col]):
                unique_vals = df[interact_col].dropna().unique()
                
                if len(unique_vals) == 2:                
                    logger.info(f'Integer interaction column "{interact_col}" detected with binary classes.')
                    pass
                else:
                    raise ValueError(f'Integer interaction column "{interact_col}" detected with more than 2 unique values: {unique_vals}. '
                                     f'Convert to categorical codes for regression.')
                
            elif pd.api.types.is_float_dtype(df[interact_col]):
                raise ValueError(f'Interaction column "{interact_col}" is float type. '
                                f'Convert to int or categorical codes for regression.')
            
            elif pd.api.types.is_string_dtype(df[interact_col]):
                raise ValueError(f'Interaction column "{interact_col}" is string type. '
                                f'Convert to categorical codes for regression.')
                

            
            # adding interaction term to the formula syntax for pyfixest
            interaction_term = f'{reg_col}:{interact_col}'
            regressors = f'{regressors} + {interaction_term}' if regressors else interaction_term

    # guard against empty regressors (e.g. include_base_regressors=False with no interaction and no trend)
    if not regressors:
        raise ValueError('No regressors specified. Provide regressor_cols, '
                         'interaction_dict, or a unit-specific trend.')

    # fixed effects
    fe = ' + '.join(fe_cols)

    # formula syntax for pyfixest: "target ~ regressors | fe1 + fe2 + ... + feN"
    formula = f"{target_col} ~ {regressors} | {fe}"

    # build vcov argument — bandwidth only applies to DK and NW
    if vcov_method in ('driscoll-kraay', 'NW'):
        vcov = {vcov_method: vcov_col, "bandwidth": bandwidth}
    else:
        vcov = {vcov_method: vcov_col}

    # model fitting
    res = pf.feols(formula=formula, data=df, vcov=vcov)

    logger.info(f'Pyfixest model fitted. Formula: {formula} | vcov: {vcov}')

    return res
