# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import re
import sys
import logging
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import pyfixest as pf
from pathlib import Path
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


class BuildPanelDF:
    
    snowmelt_months = {
    # Pacific Coast / Cascades / Sierra Nevada
    "California"    : [4, 5, 6],      # Sierra Nevada; Tulare Apr-May, San Joaquin May-Jun
    "Oregon"        : [4, 5, 6],      # Cascades + eastern OR ranges
    "Washington"    : [4, 5, 6],      # Cascades; Yakima Apr-Jun

    # Rocky Mountains / Intermountain West
    "Idaho"         : [4, 5, 6],      # Snake River basin; Apr-Jun
    "Montana"       : [4, 5, 6],      # Northern Rockies; Apr-Jun, higher elev. extends to Jul
    "Wyoming"       : [4, 5, 6],      # Wind River / Bighorn / Teton ranges
    "Colorado"      : [4, 5, 6],      # 70-80% annual runoff from Apr-Jul snowmelt
    "Utah"          : [4, 5, 6],      # Wasatch / Uinta ranges
    "Nevada"        : [4, 5],         # Humboldt + Great Basin ranges; brief Apr-May peak
    "New Mexico"    : [4, 5, 6],      # Southern Rockies / Rio Grande headwaters

    # Southwest / Lower Basin
    "Arizona"       : [],             # Lower Colorado/Gila — rain/monsoon dominated; minimal snowmelt

    # Great Plains
    "North Dakota"  : [4, 5],         # Flat terrain snowpack; rapid Apr-May melt
    "South Dakota"  : [4, 5],         # Apr-May snowmelt peak
    "Nebraska"      : [4, 5],         # Indirect snowmelt via Platte R.; low confidence
    "Kansas"        : [4, 5],         # Low/indirect snowmelt recharge; use cautiously
    "Oklahoma"      : [],             # No significant snowmelt in streamflow
    "Texas"         : [],             # Minimal — HPA recharge only 0.024 in/yr
}

    def __init__(self, n_workers=8):
        self.n_workers = n_workers

    # =============================================================================
    # Module-level helpers — must be at module level (not nested) so that
    # multiprocessing can pickle them for worker processes.
    # =============================================================================

    @staticmethod
    def _load_arr(path, no_data_value):
        """Load raster as float32 array; nodata → NaN. Returns (array, transform)."""

        if path is None or not Path(path).exists():
            return None, None
        
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            transform = src.transform
        
        arr[arr == no_data_value] = np.nan
        
        return arr, transform


    @staticmethod
    def _find_file(directory, pattern):
        """Return first file matching glob pattern, or the path itself if it's a .tif."""
        
        directory = Path(directory)
        
        if '.tif' in directory.name:
            return directory
        
        matches = list(directory.glob(pattern))
        
        return matches[0] if matches else None


    @ staticmethod
    def _parse_config(config_dict):
        """Normalise config dict values to (str path, agg_method) for pickling."""
    
        if config_dict is None:
            return {}
        
        return {k: (str(Path(v[0])), v[1]) for k, v in config_dict.items()}


    @staticmethod
    def _apply_irr_mask(arr, irr_mask, col_name, include_zero_cols):
        """Mask non-irrigated pixels; also zero-mask unless col is in include_zero_cols."""
        
        if arr.shape != irr_mask.shape:
            raise ValueError(
                f'Shape mismatch for "{col_name}": array={arr.shape}, '
                f'irr_mask={irr_mask.shape}. All rasters must share the same grid.'
            )
        
        out = arr.copy()
        out[~irr_mask] = np.nan
       
        if col_name not in include_zero_cols:
            out[out == 0] = np.nan
        
        return out


    @staticmethod
    def _run_zonal(geometries, arr, transform, stat):
        """Run zonal_stats on array; returns list of values (one per polygon)."""
        results = zonal_stats(geometries, arr, affine=transform, nodata=np.nan, stats=[stat])
        return [r.get(stat) if r.get(stat) is not None else np.nan for r in results]

    @staticmethod
    def _process_one_year(args):
        """
        Worker function: process all variables for a single year.
        Accepts a single dict of arguments (required for ProcessPoolExecutor pickling).
        Returns a partial DataFrame for that year.
        """
        year                   = args['year']
        gdf                    = args['gdf']
        irrigated_cropland_dir = Path(args['irrigated_cropland_dir'])
        monthly_data_path_dict = BuildPanelDF._parse_config(args['monthly_data_path_dict'])
        annual_data_path_dict  = BuildPanelDF._parse_config(args['annual_data_path_dict'])
        static_data_path_dict  = BuildPanelDF._parse_config(args['static_data_path_dict'])
        HUC8_name_col          = args['HUC8_name_col']
        aquifer_region_col     = args['aquifer_region_col']
        aquifer_name_col       = args['aquifer_name_col']
        state_name_col         = args['state_name_col']
        growing_season_months  = args['growing_season_months']
        include_zero_cols      = set(args['include_zero_cols'])
        no_data_value          = args['no_data_value']

        n_months = len(growing_season_months)
        all_cols = (list(annual_data_path_dict.keys()) + list(monthly_data_path_dict.keys()) +
                    list(static_data_path_dict.keys()) + ['HUC8', 'State', 'AQ_NAME', 'AQ_Region', 'year', 'month'])
        results_dict = {col: [] for col in all_cols}

        geometries = gdf.geometry

        # load irrigated mask for this year
        irr_file = BuildPanelDF._find_file(irrigated_cropland_dir, f'*{year}*.tif')
        irr_arr, _ = BuildPanelDF._load_arr(irr_file, no_data_value)
        irr_mask = (irr_arr == 1)

        # annual data (broadcast across all months)
        for col, (dir_path, agg) in annual_data_path_dict.items():
            fpath = BuildPanelDF._find_file(dir_path, f'*{year}*.tif')
            
            if fpath is None:
                results_dict[col].extend([np.nan] * len(gdf) * n_months)
                continue
            
            arr, transform = BuildPanelDF._load_arr(fpath, no_data_value)
            masked = BuildPanelDF._apply_irr_mask(arr, irr_mask, col, include_zero_cols)
            vals = BuildPanelDF._run_zonal(geometries, masked, transform, agg)
            
            results_dict[col].extend(vals * n_months)

        # static data (no irr_mask — aggregated over full HUC8)
        for col, (dir_path, agg) in static_data_path_dict.items():
            fpath = BuildPanelDF._find_file(dir_path, f'*.tif')
            
            if fpath is None:
                results_dict[col].extend([np.nan] * len(gdf) * n_months)
                continue
            
            arr, transform = BuildPanelDF._load_arr(fpath, no_data_value)
            vals = BuildPanelDF._run_zonal(geometries, arr, transform, agg)
            
            results_dict[col].extend(vals * n_months)

        # monthly data
        for month in growing_season_months:
            for col, (dir_path, agg) in monthly_data_path_dict.items():
                year_files = list(Path(dir_path).glob(f'*{year}*.tif'))
            
                if len(year_files) == 0:
                    raise ValueError(f'Monthly data missing: col="{col}", year={year}.')
            
                fpath = BuildPanelDF._find_file(dir_path, f'*{year}_{month}.tif')
                arr, transform = BuildPanelDF._load_arr(fpath, no_data_value)
                masked = BuildPanelDF._apply_irr_mask(arr, irr_mask, col, include_zero_cols)
                vals = BuildPanelDF._run_zonal(geometries, masked, transform, agg)
            
                results_dict[col].extend(vals)

            results_dict['HUC8'].extend(gdf[HUC8_name_col].values)
            results_dict['AQ_Region'].extend(gdf[aquifer_region_col].values)
            results_dict['AQ_NAME'].extend(gdf[aquifer_name_col].values)
            results_dict['State'].extend(gdf[state_name_col].values)
            results_dict['year'].extend([year] * len(gdf))
            results_dict['month'].extend([month] * len(gdf))

        return pd.DataFrame(results_dict)


    def create_monthly_panel_dataframe(
            self,
            years_list,
            HUC8_shapefile,
            HUC8_name_col,
            aquifer_region_col,
            aquifer_name_col,
            state_name_col,
            irrigated_cropland_dir,
            monthly_data_dirs,
            annual_data_dirs,
            static_data_dirs,
            streamflow_csv_path,
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
                'Water_source' : (PROJECT_ROOT / 'Data_main/rasters/USGS_GW_%/Water_source_classification/Water_source_classification.tif', 'median')
            }

            include_zero_cols = ['Precip_mm']

        :param years_list: List of years to process.
        :param HUC8_shapefile: Path to HUC8 polygon shapefile.
                                Each row = one HUC8 unit.
        :param HUC8_name_col: Shapefile column for HUC8 unit name
                            (e.g. 'HUC8' → '12050005', '11050002').
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

        include_zero_cols     = list(include_zero_cols) if include_zero_cols else []
        growing_season_months = list(growing_season_months)
        years_list            = list(years_list)

        logger.info('\n---------------------------------------------------------------')
        logger.info(f'\nStarting to compile monthly panel dataframe  |  n_workers={self.n_workers}\n')

        # -------------------------------------------------------------------------
        # path setup
        # -------------------------------------------------------------------------
        irrigated_cropland_dir = Path(irrigated_cropland_dir)
        output_csv_path        = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)


        monthly_data_path_dict = self._parse_config(monthly_data_dirs)
        annual_data_path_dict  = self._parse_config(annual_data_dirs)
        static_data_path_dict  = self._parse_config(static_data_dirs)

        # -------------------------------------------------------------------------
        # load shapefile + reproject to match raster CRS once (shared across workers)
        # -------------------------------------------------------------------------
        gdf = gpd.read_file(HUC8_shapefile)
        _ref_raster = self._find_file(irrigated_cropland_dir, f'*{years_list[0]}*.tif')
        with rio.open(_ref_raster) as _src:
            raster_crs = _src.crs
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # -------------------------------------------------------------------------
        # build per-year argument dicts — one dict passed to each worker process
        # -------------------------------------------------------------------------
        worker_args = [
            {
                'year'                  : year,
                'gdf'                   : gdf,
                'irrigated_cropland_dir': str(irrigated_cropland_dir),
                'monthly_data_path_dict': monthly_data_path_dict,
                'annual_data_path_dict' : annual_data_path_dict,
                'static_data_path_dict' : static_data_path_dict,
                'HUC8_name_col'         : HUC8_name_col,
                'aquifer_region_col'    : aquifer_region_col,
                'aquifer_name_col'      : aquifer_name_col,
                'state_name_col'        : state_name_col,
                'growing_season_months' : growing_season_months,
                'include_zero_cols'     : include_zero_cols,
                'no_data_value'         : no_data_value,
            }
            for year in years_list
        ]

        # -------------------------------------------------------------------------
        # parallel processing — one worker per year
        # -------------------------------------------------------------------------
        partial_dfs = [None] * len(years_list)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # executor.submit() launches a worker and returns a Future object (a receipt for a running job)
            # That Future becomes the dict key, and i (the index) becomes the value → {Future: i}
            # This can be later used to look up WHICH year a completed future belongs to, since as_completed()
            # returns futures in completion order (not submission order)
            future_to_idx = {executor.submit(self._process_one_year, args): i
                            for i, args in enumerate(worker_args)}
            
            for future in as_completed(future_to_idx):
                i    = future_to_idx[future]
                
                year = years_list[i]
                
                try:
                    partial_dfs[i] = future.result()
                    logger.info(f'Year {year} done.\n')
                
                except Exception as e:
                    logger.error(f'Year {year} failed: {e}')
                    raise

        # -------------------------------------------------------------------------
        # concatenate year results (partial_dfs preserves year order)
        # -------------------------------------------------------------------------
        panel_df = pd.concat(partial_dfs, ignore_index=True)

        
        # -------------------------------------------------------------------------
        # snowmelt-driven peak streamflow (max Q across snowmelt months, per HUC8-year)
        # broadcast to all months of that year for that HUC8
        # -------------------------------------------------------------------------
        stream_df = pd.read_csv(streamflow_csv_path, dtype={'HUC8': str})[['HUC8', 'year', 'Month', 'Sim_Q_naturalized_mm']]
        stream_df.loc[:, 'HUC8'] = stream_df['HUC8'].astype(str).str.zfill(8)

        # attach state and its snowmelt months to stream_df
        huc8_state = panel_df[['HUC8', 'State']].copy()
        huc8_state['HUC8'] = huc8_state['HUC8'].astype(str).str.zfill(8)
        stream_df = stream_df.merge(huc8_state, on='HUC8', how='left')
        stream_df['streamflow_months'] = stream_df['State'].map(self.snowmelt_months)
        
        # keep only rows whose month falls in that state's snowmelt window
        # (states with no snowmelt — AZ, TX, OK — map to [] and are excluded entirely)
        stream_df = stream_df[stream_df.apply(
            lambda row: isinstance(row['streamflow_months'], list) and (row['Month'] in row['streamflow_months']), axis=1)]

        # max naturalized Q across snowmelt months → one value per HUC8-year
    
        snowmelt_agg = (stream_df.groupby(['HUC8', 'year'])
                                .agg({'Sim_Q_naturalized_mm': 'max'})
                                .reset_index()
                                .rename(columns={'Sim_Q_naturalized_mm': 'Max_snowmelt_Q_mm'}))
       

        # left-merge on HUC8+year → same annual value broadcast across all months
        panel_df = panel_df.merge(snowmelt_agg, on=['HUC8', 'year'], how='left')
        
        # For HUCs with no streamflow, setting max naturalized Q as 0
        panel_df['Max_snowmelt_Q_mm'] = panel_df['Max_snowmelt_Q_mm'].fillna(0) 
        
        # -------------------------------------------------------------------------
        # adding HUC8s irrigated/non-irrigated status
        # -------------------------------------------------------------------------
        irr_status = gdf[[HUC8_name_col, 'Irrigated']]
        panel_df = panel_df.merge(irr_status, on='HUC8', how='left')
        
        # -------------------------------------------------------------------------
        # save DataFrame
        # -------------------------------------------------------------------------
        if column_rename:
            missing = [k for k in column_rename if k not in panel_df.columns]
            if missing:
                logger.warning(f'column_rename keys not found in DataFrame: {missing}')
                
            panel_df = panel_df.rename(columns=column_rename)

        panel_df.to_csv(output_csv_path, index=False)
        logger.info(f'\nRaw Panel dataframe saved → {output_csv_path}  |  shape: {panel_df.shape}\n')
        logger.info(('\n*** Further processing might be required before regression model ***\n'))
        logger.info('\n---------------------------------------------------------------')

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

    logger.info(f'STEP 2: WTD column "{WTD_col}" replaced with unit-level mean → "WTD_mean_m"\n')

    return df


def create_categorical_cols_in_df(df, categorical_config):
    """
    Create pd.Categorical columns for use as interaction variables in pyfixest regression.

    Must be run before pyfixest_fit_FE() when using categorical interaction terms (RQ1, RQ4).
    Not required for binary integer interaction columns (RQ3).

    Example inputs:
    ---------------
        categorical_config = {
            'aq_type_cat': {
                'col_name' : 'aquifer',
                'assigned_categories' : ['BR', 'CP', 'CV', 'DBA', 'HPA', 'RG', 'SRP'],
                'impose_order'    : False
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
            'aquifer_type_year'    : ['aquifer_type'],
        }
    :param year_col:  Column name for year.  Default: 'year'
    :param month_col: Column name for month. Default: 'month'

    :return: Dataframe with new FE columns and time_id column added.
    """

    for fe_col, source_cols in fe_config.items():
        df[f'{fe_col}_fe'] = df[source_cols[0]].astype(str)
        
        if len(source_cols) > 1:
            for col in source_cols[1:]:
                df[f'{fe_col}_fe'] = df[f'{fe_col}_fe'] + '_' + df[col].astype(str)

    # time_id always created — needed for DK/NW SE ordering in pyfixest (must be integer)
    df['time_id'] = df[year_col] * 12 + df[month_col]
    
    logger.info(f'STEP 4: Fixed effects columns created for: {list(fe_config.keys())}\n')
    logger.info('***** Not all FE columns will be included in the regression. *****\n')

    return df

def pyfixest_fit_FE(df, target_col, regressor_cols, fe_cols,
                    include_base_regressors=True, 
                    interaction_dict=None,
                    add_linear_trend=False,
                    unit_col=None, trend_col=None,
                    vcov_method='DK', vcov_col='time_id',
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
        
    param vcov_method: SE estimation method. Default: 'DK'.
        - 'DK'   : Driscoll-Kraay. Robust to serial correlation, spatial
                    correlation across units, and heteroskedasticity. Uses a
                    kernel-based estimator with no minimum cluster
                    requirement. Recommended for small N panels with
                    shared climate shocks (ENSO, PDO). Default.

                    Requires both vcov_col (time_id) and unit_col (panel_id):
                    within each time period, tracks whether residuals move
                    together across units (spatial correlation); across time
                    periods, tracks whether residuals are autocorrelated
                    (serial correlation). Bandwidth controls how many time
                    lags to account for.

        - 'NW'   : Newey-West. Robust to serial correlation and
                   heteroskedasticity only. Requires both vcov_col (time_id)
                   and unit_col (panel_id). Treats units as independent —
                   does not account for spatial correlation across units.
                  
        - 'CRV1' : Clustered SEs. Allows free correlation within
                   each cluster across all time periods. More flexible
                   than NW within units but ignores cross-unit
                   correlation. Asymptotic — requires large N to be
                   reliable (≥20-30 clusters).
    
    :param vcov_col: Column name for SE estimation. Usage depends on vcov_method:
        - 'DK'    : time-ordered column (e.g. 'time_id')
        - 'NW'    : time-ordered column (e.g. 'time_id')
        - 'CRV1'  : cluster column
                    ["aquifer_region", "time_id"] -> 2-way clustering
                    "aquifer_region"              -> 1-way clustering

        Default is 'time_id' for Driscoll-Kraay (DK).
    
    :param bandwidth: Number of time lags for autocorrelation correction.
        Only applies to 'DK' and 'NW'. Default: 24 (covers
        2 years of monthly autocorrelation).

    :return: Fitted pyfixest model object.
    """
    
    #----------------------------------------------------------------------------
    # base regressors
    #----------------------------------------------------------------------------
    
    # the else block handles the case (include_base_regressors=False) where interaction_dict is provided 
    # for the regressors and no separate base regressors are required
    regressors = ' + '.join(regressor_cols) if include_base_regressors else ''
        
    
    #----------------------------------------------------------------------------
    # add unit-specific linear trend if specified
    #----------------------------------------------------------------------------
    if add_linear_trend:
        if unit_col and trend_col:
            trend_term = f'{unit_col}[{trend_col}]'
            regressors = f'{regressors} + {trend_term}' if regressors else trend_term
        else:
            raise ValueError('unit_col and trend_col must be provided to add a unit-specific linear trend.')
        
    #----------------------------------------------------------------------------
    # interaction terms
    #----------------------------------------------------------------------------
    
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
            
            if not isinstance(reg_col, str) or not isinstance(interact_col, str):
                raise ValueError(f'reg_col="{reg_col}", interact_col="{interact_col}". Both must be strings corresponding to column names in the dataframe.')
           
            # adding interaction term to the formula syntax for pyfixest
            interaction_term = f'{reg_col}:{interact_col}'
            regressors = f'{regressors} + {interaction_term}' if regressors else interaction_term

    # guard against empty regressors (e.g. include_base_regressors=False with no interaction and no trend)
    if not regressors:
        raise ValueError('No regressors specified. Provide regressor_cols, '
                         'interaction_dict, or a unit-specific trend.')
        
    #----------------------------------------------------------------------------
    # fixed effects
    #----------------------------------------------------------------------------
    fe = ' + '.join(fe_cols)

    #----------------------------------------------------------------------------
    # formula syntax for pyfixest: "target ~ regressors | fe1 + fe2 + ... + feN"
    #----------------------------------------------------------------------------
    formula = f"{target_col} ~ {regressors} | {fe}"

    #----------------------------------------------------------------------------
    # vcov
    #----------------------------------------------------------------------------
    
    # build vcov argument — bandwidth only applies to DK and NW
    if vcov_method not in ('DK', 'NW', 'CRV1'):
        raise ValueError(f"Invalid vcov_method: {vcov_method}. Must be one of ('DK', 'NW', 'CRV1').")
    
    if vcov_method in ['DK', 'NW']:
        if unit_col is None:
           raise ValueError('unit_col is required for Driscoll-Kraay/Newey-West SE estimation.') 
        
        vcov = vcov_method
    
        # DK/NW requires numeric panel_id; encode unit_col to integer codes
        _panel_int_col = '__panel_id_int__'
        df[_panel_int_col] = pd.factorize(df[unit_col])[0]
        
        vcov_kwargs = {'time_id': vcov_col, 'panel_id': _panel_int_col, 'lag': bandwidth}


    else:  # CRV1
        if isinstance(vcov_col, list) and len(vcov_col) == 2:
            vcov_col = vcov_col[0] + ' + ' + vcov_col[1]  # 2-way clustering syntax
        
        vcov = {vcov_method: vcov_col}

        vcov_kwargs = None

    #----------------------------------------------------------------------------
    # model fitting
    #----------------------------------------------------------------------------
    res = pf.feols(fml=formula, data=df, vcov=vcov, vcov_kwargs=vcov_kwargs)

    logger.info(f'Pyfixest model fitted. Formula: {formula} | vcov: {vcov}')

    return res


def save_panel_model_results(
        model,
        model_name,
        output_dir,
        aquifer_state_shapefile=None,
        aquifer_region_col='aquifer_region',
        save_csv=True,
        save_shapefile=False):
    """
    Save pyfixest panel regression results as CSV and/or shapefile.

    Extracts the tidy coefficient table (estimate, SE, t-value, p-value,
    95% CI) and model-level stats (R², R² within, RMSE, N) from a fitted
    pyfixest model. Optionally joins region-specific coefficients to the
    aquifer-state shapefile for spatial export.

    Example
    -------
        save_panel_model_results(
            model=rq1,
            model_name='RQ1',
            output_dir=PROJECT_ROOT / 'Results/panel_reg',
            aquifer_state_shapefile=PROJECT_ROOT / 'Data_main/shapefiles/aquifer_state_units.shp',
            aquifer_region_col='aquifer_region',
            save_csv=True,
            save_shapefile=True,
        )

    :param model: Fitted pyfixest model object returned by pyfixest_fit_FE().
    :param model_name: Label used in the output filename and a 'model_name' column.
    :param output_dir: Directory where output files are written.
    :param aquifer_state_shapefile: Path to the aquifer-state polygon shapefile.
        Required when save_shapefile=True.
    :param aquifer_region_col: Column in the shapefile that holds aquifer_region
        labels matching those embedded in coefficient names. Default: 'aquifer_region'.
    :param save_csv: If True, save the full coefficient table as a CSV.
        Default: True.
    :param save_shapefile: If True, join region-specific coefficients to the
        shapefile geometry and save as a .shp. Requires aquifer_state_shapefile.
        Default: False.

    :return: pd.DataFrame of the full coefficient table (all rows, no geometry).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # coefficient table
    # -------------------------------------------------------------------------
    coef_df = model.tidy().reset_index()
    coef_df.insert(0, 'model_name', model_name)

    # -------------------------------------------------------------------------
    # parse coefficient names into (coef_type, aquifer_region, interaction_group)
    #
    # pattern 1 – trend            : "aquifer_region[year][T.HPA_NE]"
    #                                  → coef_type='trend', aquifer_region='HPA_NE'
    # pattern 2 – region interaction: "Precip_anomaly:aquifer_region[CP_WA]"
    #                                  → coef_type='Precip_anomaly', aquifer_region='CP_WA'
    # pattern 3 – group interaction : "Precip_anomaly:Water_source[T.0]"
    #                                  → coef_type='Precip_anomaly', interaction_group='0'
    # pattern 4 – base regressor   : "Precip_anomaly"
    #                                  → coef_type='Precip_anomaly'
    # -------------------------------------------------------------------------
    coef_col = coef_df.columns[1]  # 'Coefficient' (second column after model_name)

    trend_pat        = re.compile(rf'{re.escape(aquifer_region_col)}\[year\]\[T\.([^\]]+)\]')
    region_pat       = re.compile(rf'(\w+):{re.escape(aquifer_region_col)}\[([^\]]+)\]')
    group_pat        = re.compile(r'^([^:]+):[^[]+\[(?:T\.)?([^\]]+)\]$')

    def _parse_coef(name):
        # 1. trend term
        m = trend_pat.search(name)
        if m:
            return 'trend', m.group(1), np.nan          # coef_type, aquifer_region, interaction_group

        # 2. region-specific interaction (aquifer_region_col)
        m = region_pat.search(name)
        if m:
            return m.group(1), m.group(2), np.nan       # coef_type, aquifer_region, interaction_group

        # 3. generic group interaction  (any other VAR:COL[LEVEL])
        m = group_pat.match(name)
        if m:
            return m.group(1), np.nan, m.group(2)       # coef_type, aquifer_region, interaction_group

        # 4. base regressor or standalone term (no colon, no bracket)
        if ':' not in name and '[' not in name:
            return name, np.nan, np.nan

        return np.nan, np.nan, np.nan

    parsed = coef_df[coef_col].apply(
        lambda x: pd.Series(_parse_coef(x), index=['coef_type', 'aquifer_region', 'interaction_group'])
    )
    coef_df = pd.concat([coef_df, parsed], axis=1)
    
    # significant of note: p-value < 0.05 → significant; otherwise not significant
    coef_df['significant'] = coef_df['Pr(>|t|)'] < 0.05
    
    coef_df = coef_df.rename(columns={coef_col: 'model_term'})

    
    # -------------------------------------------------------------------------
    # save CSV
    # -------------------------------------------------------------------------
    if save_csv:
        csv_path = output_dir / f'{model_name}_results.csv'
        coef_df.to_csv(csv_path, index=False)
        logger.info(f'Results CSV saved → {csv_path}')

    # -------------------------------------------------------------------------
    # save shapefile
    # -------------------------------------------------------------------------
    if save_shapefile:
        if aquifer_state_shapefile is None:
            raise ValueError('aquifer_state_shapefile must be provided when save_shapefile=True.')

        gdf = gpd.read_file(aquifer_state_shapefile)[['State', 'AQ_code', 'AQ_State', 'AQ_Region', 'geometry']]

        spatial_df = coef_df.dropna(subset=['aquifer_region'])  # drops a row if any value in this column has NaN

        if spatial_df.empty:
            logger.warning('No region-specific coefficients found — shapefile not saved.')

        else:
            spatial_gdf = spatial_df.merge(gdf, left_on='aquifer_region', right_on='AQ_Region', how='left')
            spatial_gdf = gpd.GeoDataFrame(spatial_gdf, geometry='geometry')
            
            output_dir = output_dir / 'shapes'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            shp_path = output_dir / f'{model_name}_results.shp'
            
            spatial_gdf.to_file(shp_path)
            logger.info(f'Results shapefile saved → {shp_path}  |  rows: {len(spatial_gdf)}')

    return coef_df


def compute_IWU_from_panel_model(
        panel_df,
        results_csv,
        predictor_config,
        iwu_col='IWU_v1_mm',
        unit_col='aquifer_region',
        year_col='year',
        month_col='month',
        baseline_years=(1986, 2000),
        output_dir=None,
        model_name='RQ1'):
    """
    Compute the predictor-driven component of IWU from region-specific regression
    coefficients saved in a results CSV.

    For each predictor, the function:
      1. Optionally computes an anomaly from a raw column relative to the baseline mean.
      2. Looks up the region-specific coefficient from the results CSV.
      3. Computes: IWU_<predictor> = beta_<predictor> * predictor_value

    The total predictor-driven IWU is the sum across all predictors:
      IWU_predicted_total = sum(IWU_<predictor> for all predictors)

    predictor_config format
    -----------------------
    A dict where each key is the coef_type name as it appears in the results CSV
    (i.e., the column name used in the regression), and the value is either:

      - A raw column name (str): anomaly is computed as (raw_col - baseline_mean).
        Use this for variables like Precip_mm or Tmean_C that need detrending.

      - None: the column already exists in panel_df under the same name as the key
        and is used directly without anomaly computation.
        Use this for pre-computed variables like WinterPrecip or WTD.

    Examples:
        # Climate only (anomalies computed from raw cols):
        predictor_config = {
            'Precip_anomaly' : 'Precip_mm',   # key=coef_type, value=raw col
            'Tmean_anomaly'  : 'Tmean_C',
        }

        # Add WinterPrecip (already in panel, used as-is):
        predictor_config = {
            'Precip_anomaly' : 'Precip_mm',
            'Tmean_anomaly'  : 'Tmean_C',
            'WinterPrecip'   : None,           # used directly from panel_df
        }

    Output columns
    --------------
    For each predictor key K in predictor_config:
      - 'IWU_{K}' : predictor-driven IWU component
    Plus:
      - 'IWU_predicted_total' : sum of all components

    :param panel_df:          Monthly panel DataFrame.
    :param results_csv:       Path to model results CSV with region-specific coefficients.
    :param predictor_config:  Dict mapping coef_type → raw_col or None (see above).
    :param iwu_col:           Observed IWU column to carry through. Default: 'IWU_v1_mm'.
    :param unit_col:          Entity/region column. Default: 'aquifer_region'.
    :param year_col:          Year column. Default: 'year'.
    :param month_col:         Month column. Default: 'month'.
    :param baseline_years:    Tuple (start, end) for anomaly baseline. Default: (1986, 2000).
    :param output_dir:        Directory to save CSVs. If None, not saved.
    :param model_name:        Prefix for output filenames. Default: 'RQ1'.

    :return: Tuple (monthly_df, annual_df).
             monthly_df — one row per (region, year, month)
             annual_df  — seasonal mean per (region, year)
    """
    coef_df = pd.read_csv(results_csv)
    df      = panel_df.copy()
    base    = df[df[year_col].between(*baseline_years)]

    component_cols = []
    anomaly_cols   = []

    for coef_type, raw_col in predictor_config.items():
        if coef_type not in coef_df['coef_type'].values:
            raise ValueError(f'coef_type "{coef_type}" not found in results CSV "coef_type" column.\n' 
                             f'Available types: {coef_df["coef_type"].unique()}')
       
        # --- get region-specific coefficients for this predictor ---
        coef_series = (coef_df[coef_df['coef_type'] == coef_type]
                       .set_index(unit_col)['Estimate'])

        missing = set(df[unit_col].unique()) - set(coef_series.index)
        
        if missing:
            logger.warning(f'Regions missing coefficient for "{coef_type}": {missing}')

        # --- get predictor values ---
        if raw_col is not None:
            # compute anomaly from raw column relative to baseline mean
            clim = (base.groupby([unit_col, month_col])[raw_col]
                    .mean()
                    .rename(f'_clim_{coef_type}'))
            df = df.merge(clim, on=[unit_col, month_col])
            predictor_values = df[raw_col] - df[f'_clim_{coef_type}']
            
            col_name = coef_type if '_anomaly' in coef_type else f'{coef_type}_anomaly'
            df[col_name] = predictor_values
            anomaly_cols.append(col_name)
        
        else:
            # use the column directly from panel_df (already transformed)
            if coef_type not in df.columns:
                raise ValueError(
                    f'predictor_config: raw_col is None for "{coef_type}" but '
                    f'column "{coef_type}" not found in panel_df.'
                )
            predictor_values = df[coef_type]

        # --- compute component ---
        beta_col      = f'_beta_{coef_type}'
        component_col = f'IWU_{coef_type.replace("_anomaly", "")}'

        df[beta_col]      = df[unit_col].map(coef_series)
        df[component_col] = df[beta_col] * predictor_values
        component_cols.append(component_col)

    # total predicted IWU (sum of all components)
    df['IWU_predicted_total'] = df[component_cols].sum(axis=1)

    # -------------------------------------------------------------------------
    # build output DataFrames — drop internal helper cols
    # -------------------------------------------------------------------------
    drop_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=drop_cols)

    raw_cols = [v for v in predictor_config.values() if v is not None]

    carry_cols = [unit_col, year_col, month_col]
    for extra in ['aquifer', 'state']:
        if extra in df.columns:
            carry_cols.append(extra)
    carry_cols += raw_cols + anomaly_cols + [iwu_col] + component_cols + ['IWU_predicted_total']
    carry_cols  = list(dict.fromkeys(carry_cols))  # deduplicate, preserve order

    monthly_df = df[[c for c in carry_cols if c in df.columns]].copy()

    # -------------------------------------------------------------------------
    # ALL columns should be summed for annual aggregation
    # Anomaly sums give net annual anomaly — consistent with IWU component sums
    # Precip anomaly sum → mm/year (intuitive)
    # Tmean anomaly sum → °C·months/year (consistent, though less intuitive)

    sum_cols = [iwu_col] + component_cols + ['IWU_predicted_total'] + anomaly_cols

    agg_dict = {col: 'sum' for col in sum_cols}
        
    group_cols = [unit_col, year_col]
    for extra in ['aquifer', 'state']:
        if extra in df.columns:
            group_cols.append(extra)

    annual_df = (monthly_df.groupby(group_cols)
                   .agg(agg_dict)
                   .reset_index())

    # -------------------------------------------------------------------------
    # save
    # -------------------------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        monthly_path = output_dir / f'{model_name}_predicted_IWU_monthly.csv'
        annual_path  = output_dir / f'{model_name}_predicted_IWU_annual.csv'

        monthly_df.to_csv(monthly_path, index=False)
        annual_df.to_csv(annual_path,   index=False)

        logger.info(f'Monthly predicted IWU saved → {monthly_path}  |  rows: {len(monthly_df)}')
        logger.info(f'Annual  predicted IWU saved → {annual_path}   |  rows: {len(annual_df)}')

    return monthly_df, annual_df
