# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import re
import sys
import ast
import cloudpickle
import logging
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import pyfixest as pf
from pyfixest.estimation import demean as pyfixest_demean
from pathlib import Path
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Codes.utils.raster_ops import read_raster_arr_object
no_data_value = -9999


# CDL crop-only class codes (0 = nodata/background; non-crop land cover excluded)
# Source: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL

# Kc_mid values from FAO-56 Table 12 (Allen et al., 1998). Mid-range used where FAO-56 gives a range.
# Crops not in FAO-56 Table 12 assigned surrogate Kc from nearest botanical analog; double crops assigned averaged Kc of component crops.
# Source: https://www.fao.org/4/x0490e/x0490e0b.htm
CDL_KC_MID = {
    # ── Field crops ─────────────────────────────────────────────────────────────────────────────
    1:   1.20,   # Corn
    2:   1.18,   # Cotton
    3:   1.20,   # Rice
    4:   1.05,   # Sorghum
    5:   1.15,   # Soybeans
    6:   1.08,   # Sunflower
    10:  1.15,   # Peanuts
    11:  1.05,   # Tobacco               [surrogate: Peanuts; broad-leaved field crop, not in FAO-56]
    12:  1.15,   # Sweet Corn
    13:  1.20,   # Pop or Orn Corn       [surrogate: Corn; botanically identical crop type]
    14:  1.15,   # Mint
    # ── Small grains ───────────────────────────────────────────────────────────────────────────
    21:  1.15,   # Barley
    22:  1.15,   # Durum Wheat
    23:  1.15,   # Spring Wheat
    24:  1.15,   # Winter Wheat
    25:  1.15,   # Other Small Grains    [surrogate: Barley/Oats avg; no direct FAO-56 entry]
    27:  1.15,   # Rye  [surrogate: Barley; Rye not listed separately in FAO-56]
    28:  1.15,   # Oats
    29:  1.00,   # Millet
    30:  1.15,   # Speltz                [surrogate: Winter Wheat; ancient wheat variety]
    # ── Oilseeds ──────────────────────────────────────────────────────────────────────────────
    31:  1.08,   # Canola
    32:  1.10,   # Flaxseed
    33:  1.08,   # Safflower
    34:  1.08,   # Rape Seed
    35:  1.08,   # Mustard               [surrogate: Canola; same oilseed Brassica family]
    38:  1.08,   # Camelina              [surrogate: Canola; oilseed Brassica, similar canopy]
    39:  1.08,   # Buckwheat             [surrogate: Sunflower; broadleaf pseudo-cereal, similar canopy structure]
    # ── Forage / hay ────────────────────────────────────────────────────────────────────────────
    36:  0.95,   # Alfalfa
    37:  1.00,   # Other Hay/Non Alfalfa [surrogate: Bermuda Hay season-averaged Kc; FAO-56 Table 12]
    58:  0.90,   # Clover/Wildflowers
    59:  0.95,   # Sod/Grass Seed        [surrogate: Turf grass cool season; W-US sod crops are mostly cool-season grasses]
    60:  0.95,   # Switchgrass           [surrogate: Alfalfa; perennial grass with similar water use]
    # ── Root / tuber ───────────────────────────────────────────────────────────────────────────
    41:  1.20,   # Sugarbeets
    42:  1.15,   # Dry Beans
    43:  1.15,   # Potatoes
    45:  1.25,   # Sugarcane
    46:  1.15,   # Sweet Potatoes
    # ── Vegetables & melons ───────────────────────────────────────────────────────────────────────────
    44:  1.05,   # Other Crops           [surrogate: median field crop Kc; generic catch-all]
    47:  1.05,   # Misc Vegs & Fruits    [surrogate: median vegetable Kc; generic catch-all]
    48:  1.00,   # Watermelons
    49:  1.05,   # Onions
    50:  1.00,   # Cucumbers
    51:  1.00,   # Chick Peas
    52:  1.10,   # Lentils
    53:  1.15,   # Peas
    54:  1.15,   # Tomatoes
    55:  1.05,   # Caneberries
    56:  1.05,   # Hops
    57:  1.00,   # Herbs                 [surrogate: Garlic; small-statured aromatic crop]
    
    # ── Tree / orchard crops ───────────────────────────────────────────────────────────────────────────
    66:  0.95,   # Cherries
    67:  0.90,   # Peaches
    68:  0.95,   # Apples
    69:  0.85,   # Grapes
    70:  0.95,   # Christmas Trees       [surrogate: Cherries; managed evergreen plantation, mid-range orchard Kc]
    71:  0.95,   # Other Tree Crops      [surrogate: Pears; mid-range deciduous orchard Kc]
    72:  0.65,   # Citrus
    74:  1.10,   # Pecans                [surrogate: Walnuts; similar large deciduous nut tree]
    75:  0.90,   # Almonds
    76:  1.10,   # Walnuts
    77:  0.95,   # Pears
    # ── Specialty crops (200-series) ───────────────────────────────────────────────────────────────────────────
    204: 1.10,   # Pistachios
    205: 1.15,   # Triticale             [surrogate: Winter Wheat; wheat-rye hybrid]
    206: 1.05,   # Carrots
    207: 0.95,   # Asparagus
    208: 1.00,   # Garlic
    209: 0.85,   # Cantaloupes
    210: 0.90,   # Prunes
    211: 0.70,   # Olives
    212: 0.65,   # Oranges
    213: 1.05,   # Honeydew Melons
    214: 1.05,   # Broccoli
    215: 0.85,   # Avocados
    216: 1.05,   # Peppers
    217: 0.90,   # Pomegranates          [surrogate: Plums; similar Mediterranean deciduous fruit tree]
    218: 0.90,   # Nectarines
    219: 1.00,   # Greens                [surrogate: Lettuce; leafy vegetable]
    220: 0.90,   # Plums
    221: 0.85,   # Strawberries
    222: 0.95,   # Squash
    223: 0.90,   # Apricots
    224: 0.90,   # Vetch                 [surrogate: Clover/Wildflowers; legume forage crop]
    227: 1.00,   # Lettuce
    229: 1.00,   # Pumpkins
    242: 1.05,   # Blueberries
    243: 1.05,   # Cabbage
    244: 1.05,   # Cauliflower
    245: 1.05,   # Celery
    246: 0.90,   # Radishes
    247: 1.10,   # Turnips
    248: 1.05,   # Eggplants
    249: 1.00,   # Gourds
    250: 0.85,   # Cranberries           [surrogate: Strawberries; low-growing berry crop]
    # ── Double crops ──────────────────────────────────────────────────────────────────────────────
    26:  1.15,   # Dbl Crop WinWht/Soybeans       [avg: WinWht(1.15) + Soybeans(1.15)]
    225: 1.18,   # Dbl Crop WinWht/Corn           [avg: WinWht(1.15) + Corn(1.20)]
    226: 1.18,   # Dbl Crop Oats/Corn             [avg: Oats(1.15) + Corn(1.20)]
    228: 1.18,   # Dbl Crop Triticale/Corn        [avg: Triticale surrogate wheat(1.15) + Corn(1.20)]
    230: 1.08,   # Dbl Crop Lettuce/Durum Wht     [avg: Lettuce(1.00) + DurumWht(1.15)]
    231: 0.93,   # Dbl Crop Lettuce/Cantaloupe    [avg: Lettuce(1.00) + Cantaloupe(0.85)]
    232: 1.09,   # Dbl Crop Lettuce/Cotton        [avg: Lettuce(1.00) + Cotton(1.18)]
    233: 1.08,   # Dbl Crop Lettuce/Barley        [avg: Lettuce(1.00) + Barley(1.15)]
    234: 1.10,   # Dbl Crop Durum Wht/Sorghum     [avg: DurumWht(1.15) + Sorghum(1.05)]
    235: 1.10,   # Dbl Crop Barley/Sorghum        [avg: Barley(1.15) + Sorghum(1.05)]
    236: 1.10,   # Dbl Crop WinWht/Sorghum        [avg: WinWht(1.15) + Sorghum(1.05)]
    237: 1.18,   # Dbl Crop Barley/Corn           [avg: Barley(1.15) + Corn(1.20)]
    238: 1.17,   # Dbl Crop WinWht/Cotton         [avg: WinWht(1.15) + Cotton(1.18)]
    239: 1.17,   # Dbl Crop Soybeans/Cotton       [avg: Soybeans(1.15) + Cotton(1.18)]
    240: 1.15,   # Dbl Crop Soybeans/Oats         [avg: Soybeans(1.15) + Oats(1.15)]
    241: 1.18,   # Dbl Crop Corn/Soybeans         [avg: Corn(1.20) + Soybeans(1.15)]
    254: 1.15,   # Dbl Crop Barley/Soybeans       [avg: Barley(1.15) + Soybeans(1.15)]
}

CDL_CROP_CODES = {
    # ── Field crops ───────────────────────────────────────────────────────────
    1:  'Corn',
    2:  'Cotton',
    3:  'Rice',
    4:  'Sorghum',
    5:  'Soybeans',
    6:  'Sunflower',
    10: 'Peanuts',
    11: 'Tobacco',
    12: 'Sweet Corn',
    13: 'Pop or Orn Corn',
    14: 'Mint',
    # ── Small grains ──────────────────────────────────────────────────────────
    21: 'Barley',
    22: 'Durum Wheat',
    23: 'Spring Wheat',
    24: 'Winter Wheat',
    25: 'Other Small Grains',
    27: 'Rye',
    28: 'Oats',
    29: 'Millet',
    30: 'Speltz',
    # ── Oilseeds ──────────────────────────────────────────────────────────────
    31: 'Canola',
    32: 'Flaxseed',
    33: 'Safflower',
    34: 'Rape Seed',
    35: 'Mustard',
    38: 'Camelina',
    39: 'Buckwheat',
    # ── Forage / hay ──────────────────────────────────────────────────────────
    36: 'Alfalfa',
    37: 'Other Hay/Non Alfalfa',
    58: 'Clover/Wildflowers',
    59: 'Sod/Grass Seed',
    60: 'Switchgrass',
    # ── Root / tuber ──────────────────────────────────────────────────────────
    41: 'Sugarbeets',
    42: 'Dry Beans',
    43: 'Potatoes',
    45: 'Sugarcane',
    46: 'Sweet Potatoes',
    # ── Vegetables & melons ───────────────────────────────────────────────────
    44: 'Other Crops',
    47: 'Misc Vegs & Fruits',
    48: 'Watermelons',
    49: 'Onions',
    50: 'Cucumbers',
    51: 'Chick Peas',
    52: 'Lentils',
    53: 'Peas',
    54: 'Tomatoes',
    55: 'Caneberries',
    56: 'Hops',
    57: 'Herbs',
  
    # ── Tree / orchard crops ──────────────────────────────────────────────────
    66: 'Cherries',
    67: 'Peaches',
    68: 'Apples',
    69: 'Grapes',
    70: 'Christmas Trees',
    71: 'Other Tree Crops',
    72: 'Citrus',
    74: 'Pecans',
    75: 'Almonds',
    76: 'Walnuts',
    77: 'Pears',
    # ── Specialty crops (200-series) ──────────────────────────────────────────
    204: 'Pistachios',
    205: 'Triticale',
    206: 'Carrots',
    207: 'Asparagus',
    208: 'Garlic',
    209: 'Cantaloupes',
    210: 'Prunes',
    211: 'Olives',
    212: 'Oranges',
    213: 'Honeydew Melons',
    214: 'Broccoli',
    215: 'Avocados',
    216: 'Peppers',
    217: 'Pomegranates',
    218: 'Nectarines',
    219: 'Greens',
    220: 'Plums',
    221: 'Strawberries',
    222: 'Squash',
    223: 'Apricots',
    224: 'Vetch',
    227: 'Lettuce',
    229: 'Pumpkins',
    242: 'Blueberries',
    243: 'Cabbage',
    244: 'Cauliflower',
    245: 'Celery',
    246: 'Radishes',
    247: 'Turnips',
    248: 'Eggplants',
    249: 'Gourds',
    250: 'Cranberries',
    # ── Double crops ──────────────────────────────────────────────────────────
    26:  'Dbl Crop WinWht/Soybeans',
    225: 'Dbl Crop WinWht/Corn',
    226: 'Dbl Crop Oats/Corn',
    228: 'Dbl Crop Triticale/Corn',
    230: 'Dbl Crop Lettuce/Durum Wht',
    231: 'Dbl Crop Lettuce/Cantaloupe',
    232: 'Dbl Crop Lettuce/Cotton',
    233: 'Dbl Crop Lettuce/Barley',
    234: 'Dbl Crop Durum Wht/Sorghum',
    235: 'Dbl Crop Barley/Sorghum',
    236: 'Dbl Crop WinWht/Sorghum',
    237: 'Dbl Crop Barley/Corn',
    238: 'Dbl Crop WinWht/Cotton',
    239: 'Dbl Crop Soybeans/Cotton',
    240: 'Dbl Crop Soybeans/Oats',
    241: 'Dbl Crop Corn/Soybeans',
    254: 'Dbl Crop Barley/Soybeans',
}

# Dominant water source for irrigation in each aquifer region, based on literature review and expert knowledge.
aq_region_water_source = {
    'BR_AZ_East'      : 'Conjunctive',
    'BR_AZ_West'      : 'Conjunctive',
    'BR_NV_Central'   : 'Groundwater',
    'BR_NV_North'     : 'Conjunctive',
    'BR_NV_West'      : 'Conjunctive',
    'BR_UT_North'     : 'Surface Water',
    'BR_UT_South'     : 'Groundwater',
    'CP_OR'           : 'Surface Water',
    'CP_WA'           : 'Surface Water',
    'CV_CA_Sacramento': 'Conjunctive',
    'CV_CA_SanJoaquin': 'Conjunctive',
    'CV_CA_Tulare'    : 'Conjunctive',
    'DBA_CO'          : 'Surface Water',
    'HPA_CO'          : 'Groundwater',
    'HPA_KS_East'     : 'Groundwater',
    'HPA_KS_West'     : 'Groundwater',
    'HPA_NE'          : 'Groundwater',  
    'HPA_OK'          : 'Groundwater',  
    'HPA_TX_North'    : 'Groundwater',
    'HPA_TX_South'    : 'Groundwater',
    'RG_CO'           : 'Conjunctive',
    'RG_NM'           : 'Surface Water',
    'SRP_ID_East'     : 'Conjunctive',
    'SRP_ID_West'     : 'Surface Water',
    'Will_OR'         : 'Surface Water',
    'UCRB_CO'          : 'Surface Water',
    'UCRB_UT'          : 'Surface Water',
    'UCRB_WY'          : 'Surface Water',
}


# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


class BuildPanelDF:

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
        # monthly streamflow — actual Q for each HUC8-year-month
        # -------------------------------------------------------------------------
        monthly_q = (pd.read_csv(streamflow_csv_path, dtype={'HUC8': str})[['HUC8', 'year', 'Month', 'Sim_Q_naturalized_mm']]
                     .rename(columns={'Month': 'month', 'Sim_Q_naturalized_mm': 'Monthly_Q_mm'}))
        monthly_q['HUC8'] = monthly_q['HUC8'].astype(str).str.zfill(8)

        panel_df = panel_df.merge(monthly_q, on=['HUC8', 'year', 'month'], how='left')

        # -------------------------------------------------------------------------
        # adding HUC8s irrigated/non-irrigated status
        # -------------------------------------------------------------------------
        irr_status = gdf[[HUC8_name_col, 'Irrigated']]
        panel_df = panel_df.merge(irr_status, on='HUC8', how='left')
        
        # -------------------------------------------------------------------------
        # adding HUC8s principle water source for irrigation. This comes from
        # respective Aquifer region, therefore, at fine HUC8 scale, it may not be very accurate everywhere 
        # -------------------------------------------------------------------------
        panel_df['Water_source'] = panel_df['AQ_Region'].map(aq_region_water_source)
               
        # -------------------------------------------------------------------------
        # save DataFrame
        # -------------------------------------------------------------------------
        if column_rename:
            missing = [k for k in column_rename if k not in panel_df.columns]
            if missing:
                logger.warning(f'column_rename keys not found in DataFrame: {missing}')
                
            panel_df = panel_df.rename(columns=column_rename)

        panel_df.to_csv(output_csv_path, index=False)
        print(f'\nRaw Panel dataframe saved → {output_csv_path}  |  shape: {panel_df.shape}\n')
        print(('\n*** Further processing are required before regression model. Check panel_utils.py ***\n'))
        print('\n---------------------------------------------------------------')

        return panel_df


def clean_panel_df(df: pd.DataFrame | str,
                   nan_cols_to_consider: list,
                   years_to_consider: list = None,
                   save_path: str = None) -> pd.DataFrame:
    """

    Drops NaN rows and selects HUC8 that are irrigated.

    :param df (pd.DataFrame | str): Loaded dataframe or string path to the panel dataframe CSV.
    :param nan_cols_to_consider (list): List of columns to consider for NaN value removal.
    :param years_to_consider (list): List of years to consider. If None, all years are considered.
    :param save_path (str): Path to save the cleaned dataframe. If None, the dataframe is not saved.

    :return
        pd.DataFrame: Cleaned dataframe with NaN rows dropped and only irrigated HUC8s selected.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)

    # dropping rows with NaN values
    new_df = df.dropna(subset=nan_cols_to_consider)

    # selecting only irrigated HUC8s
    new_df = new_df[new_df['Irrigated'] == True].copy()

    # filtering by years if specified
    if years_to_consider is not None:
        new_df = new_df[new_df['year'].isin(years_to_consider)]
    
    # save cleaned dataframe if save_path is provided
    if save_path is not None:
        new_df.to_csv(save_path, index=False)
        logger.info(f'Cleaned dataframe saved → {save_path}  |  shape: {new_df.shape}\n')
    
    print(f'\nStep 1: Cleaned panel dataframe by dropping NaN rows and selecting irrigated HUC8s.')
    print('----------------'*5, '\n')
    print(f'HUC8s before cleaning the dataframe: {df["HUC8"].nunique()}')
    print(f'HUC8s after cleaning the dataframe: {new_df["HUC8"].nunique()}')
    
    return new_df


def mean_WTD_col_for_unit(df, WTD_col, unit_col='HUC8'):
    
    """
    Replace annual WTD values at the unit-level with the mean across that unit for all years.
    
    Aggregating to unit-level mean avoids endogeneity from year-to-year
    WTD changes driven by pumping history, and reflects the near-zero
    within-unit annual WTD variation relative to cross-unit gradient.

    :param df: Monthly panel dataframe.
    :param WTD_col: Column name of the WTD variable to aggregate (e.g. 'WTD_Rnd_Frst_m').
    :param unit_col: Column name for spatial unit identifier. Default: 'HUC8'.

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

    print(f'\nSTEP 2: WTD column "{WTD_col}" replaced with unit-level mean → "WTD_mean_m"')
    print('----------------'*5, '\n')
    
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

    print(f'\nSTEP 3: Categorical columns created: {list(categorical_config.keys())}')
    print('----------------'*5, '\n')

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
    
    
    print(f'\nSTEP 4: Fixed effects columns created for: {list(fe_config.keys())}')
    print('***** Not all FE columns will be included in the regression. *****')
    print('----------------'*5, '\n')

    return df


def save_load_pyfixest_model(model, save_path, save_model=True, load_model=False):
    """
    Saves or loads (or does both) a pyfixest model.

    Args:
        model (pyfixest model object): A pyfixes model object returned by pyfixest_fit_FE().
        save_path (str): Path where the model will be saved or loaded from. Must have '.pkl' extension.
        save_model (bool, optional): If True, save the model to the specified path. Defaults to True.
        load_model (bool, optional): If True, load the model from the specified path. Defaults to False.

    Returns:
        If load_model is True, returns the loaded pyfixest model object. If load_model is False, returns None. 
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_model:
        with open(save_path, 'wb') as f:
            cloudpickle.dump(model, f)

    if load_model:
        with open(save_path, 'rb') as f:
            loaded_model = cloudpickle.load(f)
        
    return loaded_model if load_model else None 


def pyfixest_fit_FE(df, target_col, regressor_cols, fe_cols,
                    include_base_regressors=True, 
                    interaction_dict=None,
                    add_linear_trend=False,
                    unit_col=None, trend_col=None,
                    vcov_method='DK', vcov_col='time_id',
                    bandwidth=24,
                    save_pyfixest_model=False, 
                    model_save_path=None):
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
    
    :param unit_col: Column name for spatial unit (e.g. 'HUC8').
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
    
    :param save_pyfixest_model: If True, saves the fitted pyfixest model object to disk using pickle.
    :param model_save_path: Path to save the pyfixest model object if save_pyfixest_model is True. 
                            Must end with .pkl.

    :return: Fitted pyfixest model object.
    """
    
    #----------------------------------------------------------------------------
    # base regressors
    #----------------------------------------------------------------------------
    
    # the else block handles the case (include_base_regressors=False) where interaction_dict is provided 
    # for the regressors and no separate base regressors are required
    if include_base_regressors is True:
        regressors = ' + '.join(regressor_cols)
        
    elif isinstance(include_base_regressors, list) and len(include_base_regressors) > 0:
        regressors = ' + '.join(include_base_regressors)
        
    else:
        regressors = ''
        
    
    #----------------------------------------------------------------------------
    # add unit-specific linear trend if specified
    #----------------------------------------------------------------------------
    if add_linear_trend:
        if unit_col and trend_col:
            trend_term = f'i({unit_col}, {trend_col})'
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
    fe_model = pf.feols(fml=formula, data=df, vcov=vcov, vcov_kwargs=vcov_kwargs)

    logger.info(f'Pyfixest model fitted. Formula: {formula} | vcov: {vcov}')

    if save_pyfixest_model:
        save_load_pyfixest_model(model=fe_model, save_path=model_save_path, 
                                 save_model=True)
        logger.info(f'Pyfixest model saved → {model_save_path}')
        
    return fe_model  
    

def save_panel_model_results(
        model,
        model_name,
        output_dir,
        incorporate_water_source_info=True,
        panel_df=None,
        shapefile=None,
        spatial_unit_col='HUC8',
        rename_sp_unit=None,
        shp_join_col=None,
        save_csv=True,
        save_shapefile=False):
    """
    Save pyfixest panel regression results as CSV and/or shapefile.

    Extracts the tidy coefficient table (estimate, SE, t-value, p-value,
    95% CI) and model-level stats (R², R² within, RMSE, N) from a fitted
    pyfixest model. Optionally joins unit-specific coefficients to a shapefile
    for spatial export. All spatial units in the shapefile are retained in the
    output (non-irrigated HUC8s appear with NaN coefficients).

    Example — aquifer scale (RQ1/RQ3)
    -----------------------------------
        save_panel_model_results(
            model=rq1,
            model_name='RQ1',
            output_dir=PROJECT_ROOT / 'Results/panel_reg',
            shapefile=PROJECT_ROOT / 'Data_main/ref_shapes/aquifers_ROI/aquifers_by_state.shp',
            spatial_unit_col='aquifer_region',
            shp_join_col='AQ_Region',
            save_csv=True,
            save_shapefile=True,
        )

    Example — HUC8 scale (RQ2)
    ----------------------------
        save_panel_model_results(
            model=rq2,
            model_name='RQ2',
            output_dir=PROJECT_ROOT / 'Results/panel_reg',
            shapefile=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8_processed.shp',
            spatial_unit_col='HUC8',
            shp_join_col='HUC8',
            save_csv=True,
            save_shapefile=True,
        )

    :param model: Fitted pyfixest model object returned by pyfixest_fit_FE().
    :param model_name: Label used in the output filename and a 'model_name' column.
    :param output_dir: Directory where output files are written.
    :param shapefile: Path to the polygon shapefile for spatial export.
        Required when save_shapefile=True.
    :param spatial_unit_col: Name of the spatial unit as it appears inside
        coefficient names (e.g. 'aquifer_region' → parses 'Precip:aquifer_region[X]';
        'HUC8' → parses 'Precip:HUC8[X]'). Default: 'aquifer_region'.
    :param rename_sp_unit: Name to rename the spatial unit column to in the output.
    :param shp_join_col: Column in the shapefile used as the join key
        (e.g. 'AQ_Region' for aquifer shapefile, 'HUC8' for HUC8 shapefile).
        Defaults to spatial_unit_col if not provided.
    :param save_csv: If True, save the full coefficient table as a CSV.
        Default: True.
    :param save_shapefile: If True, join unit-specific coefficients to the
        shapefile geometry and save as a .shp. Requires shapefile.
        Default: False.

    :return: pd.DataFrame of the full coefficient table (all rows, no geometry).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _shp_join_col = shp_join_col if shp_join_col is not None else spatial_unit_col

    # -------------------------------------------------------------------------
    # coefficient table
    # -------------------------------------------------------------------------
    coef_df = model.tidy().reset_index()
    coef_df.insert(0, 'model_name', model_name)

    # -------------------------------------------------------------------------
    # parse coefficient names into (coef_type, spatial_unit, interaction_group)
    #
    # pattern 1 – trend            : "C(HUC8)[13020101]:year"  (from i(HUC8, year) syntax)
    #                                  → coef_type='trend', spatial_unit='13020101'
    # pattern 2 – unit interaction : "Precip_anomaly:aquifer_region[CP_WA]"  or  "Precip_anomaly:HUC8[13020101]"
    #                                  → coef_type='Precip_anomaly', spatial_unit='CP_WA'
    # pattern 3 – group interaction: "Precip_anomaly:Water_source[T.0]"
    #                                  → coef_type='Precip_anomaly', interaction_group='0'
    # pattern 4 – base regressor   : "Precip_anomaly"
    #                                  → coef_type='Precip_anomaly'
    # -------------------------------------------------------------------------
    coef_col = coef_df.columns[1]  # 'Coefficient' (second column after model_name)

    trend_pat  = re.compile(rf'C\({re.escape(spatial_unit_col)}\)\[([^\]]+)\]:year')
    unit_pat   = re.compile(rf'(\w+):{re.escape(spatial_unit_col)}\[([^\]]+)\]')
    group_pat  = re.compile(r'^([^:]+):[^[]+\[(?:T\.)?([^\]]+)\]$')

    def _parse_coef(name):
        # 1. trend term
        m = trend_pat.search(name)
        if m:
            return 'trend', m.group(1)

        # 2. unit-specific interaction
        m = unit_pat.search(name)
        if m:
            return m.group(1), m.group(2)

        # 3. generic group interaction (any other VAR:COL[LEVEL])
        m = group_pat.match(name)
        if m:
            return m.group(1), m.group(2)

        # 4. base regressor or standalone term
        if ':' not in name and '[' not in name:
            return name, np.nan

        return np.nan, np.nan

    sp_unit_name = rename_sp_unit if rename_sp_unit is not None else 'sp_unit'

    parsed = coef_df[coef_col].apply(
        lambda x: pd.Series(_parse_coef(x), index=['coef_type', sp_unit_name])
    )
    coef_df = pd.concat([coef_df, parsed], axis=1)
    coef_df[sp_unit_name] = coef_df[sp_unit_name].where(coef_df[sp_unit_name].notna(), other=np.nan)

    coef_df['SIG'] = coef_df['Pr(>|t|)'] < 0.05

    coef_df = coef_df.rename(columns={coef_col: 'model_term'})

    # -------------------------------------------------------------------------
    # pivot to wide format
    # -------------------------------------------------------------------------
    unit_rows = coef_df[coef_df[sp_unit_name].notna()]
    unit_rows = unit_rows.rename(columns={'Std. Error': 'SE', 't value': 't-stat', 'Pr(>|t|)': 'p-value'})
    value_cols = ['Estimate', 'SE', 't-stat', 'p-value', '2.5%', '97.5%', 'SIG']
       
    wide_df = unit_rows.pivot_table(
        index = [sp_unit_name],
        columns = 'coef_type',
        values = value_cols
    )
    
    # The pivoted df has multi-index columns
    # Converting them back to single level with format "{value_col}_{coef_type}"
    coef_rename_dict = {'Monthly_Q_mm': 'Q', 'Precip_mm': 'P', 'Tmean_C': 'T', 'trend': 'trend'}
    
    wide_df.columns = [f"{coef_rename_dict[coef]}_{stat}" for stat, coef in wide_df.columns]
    wide_df = wide_df.reset_index()  # bring HUC8 back as a regular column

    # -------------------------------------------------------------------------
    # attach dominant Water_source per spatial unit from panel_df
    # -------------------------------------------------------------------------
    if incorporate_water_source_info and panel_df is not None:
        ws_map = panel_df.drop_duplicates(subset=[spatial_unit_col]).set_index(spatial_unit_col)['Water_source']
        ws_map.index = ws_map.index.astype(str)
        wide_df['Water_source'] = wide_df[sp_unit_name].astype(str).map(ws_map)

    # -------------------------------------------------------------------------
    # attach State and AQ_Region via shapefile lookup
    # -------------------------------------------------------------------------
    # Only initialise columns that aren't already the spatial unit index
    for col in ['State', 'AQ_Region']:
        if col not in wide_df.columns:
            wide_df[col] = np.nan

    if shapefile is not None:
        shp_gdf    = gpd.read_file(shapefile)
        extra_cols = [c for c in ['State', 'AQ_Region'] if c != _shp_join_col and c in shp_gdf.columns]
        shp_lookup = shp_gdf[[_shp_join_col] + extra_cols].drop_duplicates()
        shp_lookup[_shp_join_col] = shp_lookup[_shp_join_col].astype(str)
        wide_df[sp_unit_name] = wide_df[sp_unit_name].astype(str)

        rename_map = {_shp_join_col: sp_unit_name}
        if 'State'     in extra_cols: rename_map['State']     = '_State'
        if 'AQ_Region' in extra_cols: rename_map['AQ_Region'] = '_AQ_Region'

        wide_df = wide_df.merge(shp_lookup.rename(columns=rename_map), on=sp_unit_name, how='left')

        if '_State' in wide_df.columns:
            wide_df['State']     = wide_df['_State'].where(wide_df['_State'].notna(), other=np.nan)
            wide_df = wide_df.drop(columns=['_State'])
        if '_AQ_Region' in wide_df.columns:
            wide_df['AQ_Region'] = wide_df['_AQ_Region'].where(wide_df['_AQ_Region'].notna(), other=np.nan)
            wide_df = wide_df.drop(columns=['_AQ_Region'])


    # organizing the columns
    index_cols = [sp_unit_name, 'State', 'AQ_Region', 'Water_source'] if incorporate_water_source_info and panel_df is not None else [sp_unit_name, 'State', 'AQ_Region']
    coef_cols = [col for col in wide_df.columns if col not in index_cols]
    
    coef_cols_sorted = sorted(coef_cols, key=lambda x: x.split('_')[0])
    
    wide_df = wide_df[index_cols + coef_cols_sorted]
    
    # -------------------------------------------------------------------------
    # save CSV
    # -------------------------------------------------------------------------
    if save_csv:
        csv_path = output_dir / f'{model_name}_results.csv'
        wide_df.to_csv(csv_path, index=False)
        logger.info(f'Results CSV saved → {csv_path}')

    # -------------------------------------------------------------------------
    # save shapefile
    # -------------------------------------------------------------------------
    if save_shapefile:
        if shapefile is None:
            raise ValueError('shapefile must be provided when save_shapefile=True.')

        gdf = gpd.read_file(shapefile)
        gdf[_shp_join_col] = gdf[_shp_join_col].astype(str)
        wide_df[sp_unit_name] = wide_df[sp_unit_name].astype(str)

        if wide_df.empty:
            logger.warning('No unit-specific coefficients found — shapefile not saved.')

        else:
            # drop columns already in gdf to avoid _x/_y duplicates after merge
            dup_cols = [c for c in ['State', 'AQ_Region'] if c in wide_df.columns and c in gdf.columns]
            wide_shp = wide_df.drop(columns=dup_cols)

            # left join from shapefile so ALL HUC8s appear (non-modeled get NaN coefficients)
            spatial_gdf = gdf.merge(wide_shp, left_on=_shp_join_col, right_on=sp_unit_name, how='left')
            spatial_gdf = spatial_gdf.drop(columns=[sp_unit_name], errors='ignore')
            spatial_gdf = gpd.GeoDataFrame(spatial_gdf, geometry='geometry')

            # rename columns that exceed ESRI shapefile 10-char field name limit
            shp_col_rename = {
                'model_name'  : 'mod_name',
                'model_term'  : 'mod_term',
                'Std. Error'  : 'Std_Err',
                'Pr(>|t|)'    : 'p_value',
                'significant' : 'sig',
                'coef_type'   : 'coef_type',
                'interaction_group': 'int_grp',
                'Water_source': 'Water_SR',
            }
            shp_col_rename = {k: v for k, v in shp_col_rename.items() if k in spatial_gdf.columns}
            spatial_gdf = spatial_gdf.rename(columns=shp_col_rename)

            shapes_dir = output_dir / 'shapes'
            shapes_dir.mkdir(parents=True, exist_ok=True)

            shp_path = shapes_dir / f'{model_name}_results.shp'
            spatial_gdf.to_file(shp_path)
            logger.info(f'Results shapefile saved → {shp_path}  |  rows: {len(spatial_gdf)}')

    return wide_df


def get_demeaned_vars(panel_df, predictors_list):
    
    """
    Demean the raw predictors by both HUC8×month and State×year fixed effects using 
    pyfixest's alternating projections algorithm, leaving only the within-watershed, within-state
    climate anomaly — exactly as the regression model saw it.
    
    WHY NOT simple two-step group-mean subtraction?
    ------------------------------------------------
    The naive approach (X - HUC8_month_mean - State_year_mean + grand_mean)
    is only exact for balanced, orthogonal panels. Here, HUC8s are nested
    within states and the panel is unbalanced (dropped some HUC8s based on annual irrigation status)
    This means the two FE groups share overlapping variance — satisfying one group's zero-mean
    condition in one step disturbs the other group's condition, because the
    State×year mean is an unequal mix of HUC8×month groups across years.
    The result is that "demeaned" predictors still carry residual State×year
    or HUC8×month variation, inflating the estimated IWU_climate values.
    
    HOW alternating projections fixes this (using alternating projection - Gausse-Seidel algorithm):
    ------------------------------------------------
    The algorithm iterates between the two demeaning steps until both
    conditions hold simultaneously:
      Round 1:  R1 = X  - mean(X  | HUC8, month)    → HUC8×month means = 0
                R2 = R1 - mean(R1 | State, year)    → State×year means = 0
                                                       (but HUC8×month slightly broken)
      Round 2:  R3 = R2 - mean(R2 | HUC8, month)    → HUC8×month means = 0 again
                R4 = R3 - mean(R3 | State, year)    → State×year means = 0 again
                                                       (HUC8×month even less broken)
      ...repeat until max(|R_k+1 - R_k|) < tol (default 1e-8)
    
    Each pass, the residual contamination between the two FE groups shrinks
    because each correction is proportional to the previous violation.
    At convergence, both zero-mean conditions hold simultaneously — this is
    the exact projection used internally by pyfixest during regression, so
    the demeaned predictors here are fully consistent with the fitted model.
    
    :param panel_df:        Monthly panel DataFrame with columns HUC8, State, AQ_Region,
                            Water_source, year, month, Irrigated, and all predictors_list cols.
    :param predictors_list: Raw predictor column names (e.g. ['Precip_mm', 'Tmean_C']).
    
    :return: DataFrame with the same structure as panel_df but only the predictors_list columns.
    """
    
    pdf = panel_df.copy()

    # ------------------------------------------------------------------------
    # prepare predictor columns
    # ------------------------------------------------------------------------
    useful_cols = ['HUC8', 'State', 'AQ_Region', 'Water_source', 'year', 'month', 'Irrigated'] + predictors_list

    pdf = pdf[useful_cols].copy()

    # -------------------------------------------------------------------------
    # Two-way FE demeaning via pyfixest's alternating projections algorithm
    # -------------------------------------------------------------------------

    # Rows with any NaN in predictors cannot be demeaned — exclude them, demean
    # the valid rows, then write results back into their original positions.
    valid_mask = pdf[predictors_list].notna().all(axis=1)
    pdf_valid  = pdf[valid_mask].copy().reset_index(drop=True)

    # Build integer-coded FE arrays (pyfixest demean() requires integer indices)
    # Concatenate HUC8+month and State+year into strings first so each unique
    # (HUC8, month) pair and each unique (State, year) pair gets one integer label.
    huc8_month_codes = pd.Categorical(
        pdf_valid['HUC8'].astype(str) + '_' + pdf_valid['month'].astype(str)
    ).codes
    
    # state_year_codes = pd.Categorical(
    #     pdf_valid['State'].astype(str) + '_' + pdf_valid['year'].astype(str)
    # ).codes
    
    year_codes = pd.Categorical(pdf_valid['year'].astype(str)).codes

    flist   = np.column_stack([huc8_month_codes,
                               year_codes,
                            #    state_year_codes
                               ]).astype(int)
    weights = np.ones(len(pdf_valid), dtype=float)

    X_raw = pdf_valid[predictors_list].to_numpy(dtype=float)  # shape: (n_valid_rows, n_predictors)

    X_demeaned, converged = pyfixest_demean(X_raw, flist, weights)  # shape: (n_valid_rows, n_predictors)

    if not converged:
        logger.warning('pyfixest demean() did not converge — results may be inaccurate.')

    # Write demeaned values back; invalid rows stay NaN (handled downstream as 0)
    updated_pdf = pdf.copy()
    for i, col in enumerate(predictors_list):
        updated_pdf.loc[valid_mask, col] = X_demeaned[:, i]
        updated_pdf.loc[~valid_mask, col] = np.nan
        
    return updated_pdf


def compute_CIs_of_total_climate_IWU(trained_fe_model_pkl, demeaned_df, N_draws=2000):
    
    # Helper function to parse HUC8 name
    def parse_huc8_num(coef_name):
        
        m = re.search(r'HUC8\[([^\]]+)\]', coef_name)
        
        return m.group(1) if m else None
    
    # # ----------- Calculating the total climate-driven IWU and its CIs ---------------------------------
    # Here, we will need to consider the covariance between the precip and tmean coefficients to get 
    # accurate CIs for the combined climate effect.
    
    # Loading model
    trained_fe_model = save_load_pyfixest_model(model=None, 
                                                save_path=trained_fe_model_pkl, 
                                                save_model=False, 
                                                load_model=True)
    
    
    # Extracting covariance matrix and coefficient values
    coef_names = trained_fe_model.coef().index.tolist()
    
    P_indices = [i for i, name in enumerate(coef_names) if name.startswith('Precip')]
    T_indices = [i for i, name in enumerate(coef_names) if name.startswith('Tmean')]
    PT_indices = P_indices + T_indices
    
    coef_PT_names = [coef_names[i] for i in PT_indices]
    
    coef_values = trained_fe_model.coef().values
    beta_PT = [coef_values[i] for i in PT_indices]
        
    vcov_mat = trained_fe_model._vcov
    vcov_PT = vcov_mat[np.ix_(PT_indices, PT_indices)]  # selecting the submatrix for P and T coefficients
    
    # creating a normal distribtuion of betas for each HUC8 and draw N_draws sample from them
    beta_draws = np.random.multivariate_normal(mean=beta_PT, cov=vcov_PT, size=N_draws)  # shape: (N_draws, len(PT_indices))

    # Building two lookup dictionaries to map HUC8 numbers to their corresponding column indices 
    # in beta_draws for P and T coefficients
    P_col_for_huc8 = {}
    T_col_for_huc8 = {}
    
    for k, name in enumerate(coef_PT_names):
        huc8_num = parse_huc8_num(name)
        
        if huc8_num is None:
            continue
        
        if name.startswith('Precip'):
            P_col_for_huc8[huc8_num] = k  # column K of beta_draws is β_P for this HUC8; we know that because
                                          # coef_PT_names and beta_P share the same order as PT_indices.
                                          # beta_PT was later used to create beta_draws, so the order is preserved.
                                          
        elif name.startswith('Tmean'):
            T_col_for_huc8[huc8_num] = k  # column K of beta_draws is β_T for this HUC8

    # Now, we are going through each HUC8 in the demeaned_df and finding the 
    # corresponding column indices for that huc8 in beta_draws for both P and T coefficients
    # using the lookup dictionaries we just created.
    huc8_strs = demeaned_df['HUC8'].astype(str).values
    idx_P = np.array([P_col_for_huc8[huc8] for huc8 in huc8_strs])
    idx_T = np.array([T_col_for_huc8[huc8] for huc8 in huc8_strs])
    
    # extracting the demeand P and T precitors
    X_P = demeaned_df['Precip_mm'].values  # shape: (n_rows,)
    X_T = demeaned_df['Tmean_C'].values     # shape: (n_rows,)  
    
    # computing IWU_climate in a vectorized format
    
    # beta_draws shape: (N_draws, len(PT_indices))
    # beta_draws[:, idx_P] shape: (N_draws, N_rows) — for each draw, pulls the right β_P for each row
    # X_P shape: (N_rows,) — broadcasts across the draw axis
    IWU_climate_draws = beta_draws[:, idx_P] * X_P + beta_draws[:, idx_T] * X_T  # shape: (N_draws, N_rows) = (N_draws, ~235,000 if 576 HUC8s × 12 months × 34 years)
    
    # Building a dataframe where each column is one draw's monthly IWU
    draws_df = pd.DataFrame(
        IWU_climate_draws.T,  # transpose to have rows as months and columns as draws
        columns=[f'draw_{i}' for i in range(N_draws)]
    )
    
    draws_df['HUC8'] = demeaned_df['HUC8'].values
    draws_df['year'] = demeaned_df['year'].values
    draws_df['month'] = demeaned_df['month'].values
    
    draws_col = [col for col in draws_df.columns if col.startswith('draw_')]
    draws_only = draws_df[draws_col].values
    
    draws_df['IWU_climate_2.5%'] = np.percentile(draws_only, 2.5, axis=1)
    draws_df['IWU_climate_97.5%'] = np.percentile(draws_only, 97.5, axis=1)
    
    monthly_df = draws_df[['HUC8', 'year', 'month', 'IWU_climate_2.5%', 'IWU_climate_97.5%']].copy()
    
    # ------- Sum monthly → annual WITHIN each draw, then take quantiles -------------
    annual_df = draws_df.groupby(['HUC8', 'year'])[draws_col].sum().reset_index()
    annual_draws_only = annual_df[draws_col].values     # shape (N_HUC8·N_years, N_draws)
    annual_df['IWU_climate_2.5%']  = np.percentile(annual_draws_only,  2.5, axis=1)
    annual_df['IWU_climate_97.5%'] = np.percentile(annual_draws_only, 97.5, axis=1)
    
    annual_df = annual_df[['HUC8', 'year', 'IWU_climate_2.5%', 'IWU_climate_97.5%']].copy()

    return monthly_df, annual_df
    
    

def compute_climate_trend_IWU_fe_model(
        trained_fe_model_pkl,
        panel_df,
        predictors_list,
        coef_csv,
        model_name,
        output_dir):
    """
    Compute climate-driven (P, T) and trend-driven IWU components from HUC8-specific
    FE model coefficients.

    Applies two-way FE demeaning (HUC8×month and State×year) to raw predictors, then
    multiplies by HUC8-specific coefficients from the results CSV. Trend-driven IWU is
    computed as δ[j] × (year − baseline_year) at the monthly level and summed annually.
    Requires a results CSV from a model fitted with add_linear_trend=True.

    :param panel_df:        Monthly panel DataFrame with columns HUC8, State, AQ_Region,
                            Water_source, year, month, Irrigated, and all predictors_list cols.
    :param predictors_list: Raw predictor column names (e.g. ['Precip_mm', 'Tmean_C']).
    :param coef_csv:        Path to results CSV from save_panel_model_results() with trend cols.
    :param model_name:      Prefix for output filenames.
    :param output_dir:      Directory where monthly and annual CSVs are saved.

    :return: Tuple (monthly_df, annual_df).
    """
    coef_df = pd.read_csv(coef_csv)
    pdf = panel_df.copy()

    # ------------------------------------------------------------------------
    # prepare predictor columns
    # ------------------------------------------------------------------------
    useful_cols = ['HUC8', 'State', 'AQ_Region', 'Water_source', 'year', 'month', 'Irrigated'] + predictors_list

    pdf = pdf[useful_cols].copy()

    # -------------------------------------------------------------------------
    # Two-way FE demeaning via pyfixest's alternating projections algorithm
    # -------------------------------------------------------------------------
    updated_pdf = get_demeaned_vars(panel_df=pdf, predictors_list=predictors_list)
        
    # -------------------------------------------------------------------------
    # Brining-in estimated coeffcients to the dataframe
    # -------------------------------------------------------------------------
    coef_df = coef_df.drop(columns=['State', 'AQ_Region', 'Water_source'], errors='ignore')
    aligned_df = updated_pdf.merge(coef_df, on='HUC8', how='left')

    # -------------------------------------------------------------------------
    # calculate climate-driven IWU components and total predicted IWU
    # -------------------------------------------------------------------------
    for col in predictors_list:
        if 'precip' in col.lower():
            aligned_df['IWU_precip'] = aligned_df[f'P_Estimate'] * aligned_df[col]
            aligned_df['IWU_precip_2.5%'] = aligned_df['IWU_precip'] - 1.96 * aligned_df['P_SE'] * aligned_df[col].abs()
            aligned_df['IWU_precip_97.5%'] = aligned_df['IWU_precip'] + 1.96 * aligned_df['P_SE'] * aligned_df[col].abs()
            
        elif 'tmean' in col.lower():
            aligned_df['IWU_tmean'] = aligned_df[f'T_Estimate'] * aligned_df[col]
            aligned_df['IWU_tmean_2.5%'] = aligned_df['IWU_tmean'] - 1.96 * aligned_df['T_SE'] * aligned_df[col].abs()
            aligned_df['IWU_tmean_97.5%'] = aligned_df['IWU_tmean'] + 1.96 * aligned_df['T_SE'] * aligned_df[col].abs()

        elif 'monthly_q' in col.lower():
            aligned_df['IWU_monthly_q'] = aligned_df[f'Q_Estimate'] * aligned_df[col]
            aligned_df['IWU_monthly_q_2.5%'] = aligned_df['IWU_monthly_q'] - 1.96 * aligned_df['Q_SE'] * aligned_df[col].abs()
            aligned_df['IWU_monthly_q_97.5%'] = aligned_df['IWU_monthly_q'] + 1.96 * aligned_df['Q_SE'] * aligned_df[col].abs()

    # IWU trend calculation
    baseline_year = aligned_df['year'].min()
    yrs_since = aligned_df['year'] - baseline_year
    aligned_df['IWU_trend']      = yrs_since * aligned_df['trend_Estimate']
    aligned_df['IWU_trend_2.5%']  = aligned_df['IWU_trend'] - 1.96 * aligned_df['trend_SE'] * yrs_since
    aligned_df['IWU_trend_97.5%'] = aligned_df['IWU_trend'] + 1.96 * aligned_df['trend_SE'] * yrs_since
    
    #######################################################################################################
    # Now, we need to calculate the CIs for the total climate-driven IWU (P + T) considering the covariance 
    # between P and T coefficients. This process in itself is a bit involved, so I have implemented it in 
    # a separate function called compute_CIs_of_total_climate_IWU().
    IWU_climate_monthly_df, IWU_climate_annual_df = \
        compute_CIs_of_total_climate_IWU(trained_fe_model_pkl, 
                                        demeaned_df=aligned_df, 
                                        N_draws=2000)
    
    aligned_df['IWU_climate'] = aligned_df[['IWU_precip', 'IWU_tmean']].sum(axis=1)
    aligned_df = aligned_df.merge(IWU_climate_monthly_df, on=['HUC8', 'year', 'month'], how='left')

    monthly_df = aligned_df.copy()
    
    # -------------------------------------------------------------------------
    # aggregate to annual dataframe
    # -------------------------------------------------------------------------
    annual_df = (monthly_df.groupby(['HUC8', 'year'])
                 .agg({
                     'IWU_precip': 'sum',
                     'IWU_precip_2.5%': 'sum',
                     'IWU_precip_97.5%': 'sum',
                     'IWU_tmean': 'sum',
                     'IWU_tmean_2.5%': 'sum',
                     'IWU_tmean_97.5%': 'sum',
                     'IWU_climate': 'sum',
                     'IWU_trend': 'sum',
                     'IWU_trend_2.5%': 'sum',
                     'IWU_trend_97.5%': 'sum',
                     'State': 'first',
                     'AQ_Region': 'first',
                     'Water_source': 'first',
                     'Irrigated': 'first'
                 })
                 .reset_index())
    
    annual_df = annual_df.merge(IWU_climate_annual_df, on=['HUC8', 'year'], how='left')

    # -------------------------------------------------------------------------
    # save
    # -------------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = output_dir / f'{model_name}_predicted_IWU_monthly.csv'
    annual_path  = output_dir / f'{model_name}_predicted_IWU_annual.csv'

    monthly_df.to_csv(monthly_path, index=False)
    annual_df.to_csv(annual_path,   index=False)

    logger.info(f'Monthly predicted IWU saved → {monthly_path}')
    logger.info(f'Annual  predicted IWU saved → {annual_path}')

    return monthly_df, annual_df


def extract_info_30m_cdl(huc8_shape, conus_cdl_dir, output_dir, skip_processing=False):
    """
    Extract crop pixel counts and fractional shares per HUC8 basin from 30m CONUS CDL rasters.

    For each year (2008–2023), reads the CONUS CDL raster, masks out non-crop pixels using
    CDL_CROP_CODES, and runs zonal statistics over irrigated HUC8 basins to count pixels per
    crop class. Results are saved as a CSV with one row per HUC8-year containing crop counts
    and fractional shares of each crop class out of total crop pixels.

    :param huc8_shape: Filepath to HUC8 shapefile. Must contain an 'Irrigated' boolean column.
    :param conus_cdl_dir: Directory containing yearly CONUS CDL rasters (one .tif per year).
    :param output_dir: Directory where the output CSV will be saved.
    :param skip_processing: Set to True to skip this step.

    :return: None
    """
    if skip_processing:
        return None

    conus_cdl_dir = Path(conus_cdl_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load HUC8 shapefile and keep only irrigated basins
    huc8 = gpd.read_file(huc8_shape)
    huc8 = huc8[huc8['Irrigated'] == True]

    # reproject HUC8 to CDL CRS once (CDL is always EPSG:5070 Albers; HUC8 is typically EPSG:4269)
    first_cdl = sorted(conus_cdl_dir.glob('*.tif'))[0]
    with rio.open(first_cdl) as src:
        cdl_crs = src.crs
    if huc8.crs != cdl_crs:
        logger.info(f'Reprojecting HUC8 from {huc8.crs} to CDL CRS {cdl_crs}...')
        huc8 = huc8.to_crs(cdl_crs)

    years_df = []

    for year in range(2008, 2024):
        logger.info(f'Extracting info from 30m CDL for year {year}...')

        # find the CDL raster for this year
        cdl_file = list(conus_cdl_dir.glob(f'*{year}*.tif'))[0]

        with rio.open(cdl_file) as src:
            affine = src.transform

            # read raw CDL array and ensure background pixels are 0
            cdl_arr = src.read(1)
            if src.nodata is not None:
                cdl_arr[cdl_arr == src.nodata] = 0

            # mask out non-crop pixels (set to 0 = nodata) keeping only CDL_CROP_CODES
            crop_pixels = list(CDL_CROP_CODES.keys())
            cdl_arr_filtered = np.where(np.isin(cdl_arr, crop_pixels), cdl_arr, 0)

        # count crop pixels per class within each HUC8 (categorical = pixel counts per class code)
        stats = zonal_stats(huc8,
                            cdl_arr_filtered,
                            affine=affine,
                            categorical=True,
                            nodata=0)

        records = []

        for huc_feat, stat in zip(huc8.itertuples(), stats):
            # keep only valid integer crop class keys
            crop_counts = {
                k: v for k, v in stat.items()
                if str(k).isdigit() and v is not None
            }

            # skip HUC8s with no crop pixels
            total = sum(crop_counts.values())
            if total == 0:
                continue

            # compute fractional share of each crop class out of total crop pixels
            crop_shares = {k: v / total for k, v in crop_counts.items()}

            records.append({
                'HUC8':        huc_feat.HUC8,
                'Year':        year,
                'Crop_Counts': crop_counts,
                'Crop_Shares': crop_shares
            })

        years_df.append(pd.DataFrame(records))

    # combine all years into one DataFrame
    df_all = pd.concat(years_df, ignore_index=True)
    
    # save to csv
    output_csv = output_dir / 'HUC8_CDL_crop_counts_shares.csv'
    df_all.to_csv(output_csv, index=False)
    logger.info(f'CDL crop counts and shares saved to {output_csv}')


def build_huc8_postprocess_df(huc8_cdl_csv, huc8_shape,
                             pet_p_corr_raster, wtd_raster,
                             irrigated_cropland_dir, gs_kc_dir,
                             years, output_csv, skip_processing=False):
    """
    Build a full HUC8 × year panel with five blocks of variables:
      - Block 1 (CDL, 2008–2023): top 3 crops, their shares, Kc values, and weighted Kc.
      - Block 2 (PET_P corr, all years): mean Pearson correlation between P and PET for
        irrigated pixels within each HUC8, using the annual irrigated cropland mask.
      - Block 3 (WTD, all years): mean water table depth for irrigated pixels within each HUC8,
        using the annual irrigated cropland mask.
      - Block 4 (Water source, all years): dominant irrigation water source per AQ_Region,
        mapped from aq_region_water_source dictionary.
      - Block 5 (ET_frac, all years): mean growing-season Kc for irrigated pixels within each
        HUC8, derived from annual GS_Kc rasters masked with the irrigated cropland raster.

    CDL columns are NaN for years before 2008 (CDL not available); all other columns
    have full coverage across all years in `years` (where rasters exist).

    Note: All rasters must be on the same grid/resolution.

    :param huc8_cdl_csv: Filepath to input CSV from extract_info_30m_cdl().
    :param huc8_shape: Filepath to HUC8 shapefile. Must contain 'HUC8' and 'Irrigated' columns.
    :param pet_p_corr_raster: Filepath to static PET_P Pearson correlation raster.
    :param wtd_raster: Filepath to static mean water table depth raster.
    :param irrigated_cropland_dir: Directory with annual irrigated cropland rasters
                                   named 'Irrigated_cropland_{year}.tif' (1 = irrigated).
    :param gs_kc_dir: Directory with annual growing-season Kc rasters named 'GS_Kc_{year}.tif'.
    :param years: List of years for the full panel (e.g. list(range(1986, 2024))).
    :param output_csv: Filepath for the output panel CSV.
    :param skip_processing: Set to True to skip this step.

    :return: None
    """
    if skip_processing:
        return None

    logger.info('Building HUC8 panel: CDL / PET_P correlation / WTD...')
    logger.info('-----------------------------------------------------\n')

    output_csv = Path(output_csv)
    irrigated_cropland_dir = Path(irrigated_cropland_dir)

    # load irrigated HUC8 basins
    huc8 = gpd.read_file(huc8_shape)
    huc8['HUC8'] = huc8['HUC8'].astype(str)
    huc8 = huc8[huc8['Irrigated'] == True].reset_index(drop=True)

    # create full panel skeleton: all HUC8s × all years, with State and AQ_Region
    huc8_meta = huc8[['HUC8', 'State', 'AQ_Region']].drop_duplicates()
    panel_df = pd.DataFrame(
        [(huc, yr) for huc in huc8['HUC8'] for yr in years],
        columns=['HUC8', 'Year']
    )
    panel_df = panel_df.merge(huc8_meta, on='HUC8', how='left')

    # ── Block 1: CDL (2008–2023) ───────────────────────────────────────────────────────────────
    logger.info('Block 1: processing CDL crop info (2008-2023)...\n')

    cdl_df = pd.read_csv(huc8_cdl_csv)
    cdl_df['HUC8'] = cdl_df['HUC8'].astype(str)
    cdl_df['Crop_Counts'] = cdl_df['Crop_Counts'].apply(ast.literal_eval)
    cdl_df['Crop_shares'] = cdl_df['Crop_Shares'].apply(ast.literal_eval)

    cdl_df['Top_Crop_code']    = cdl_df['Crop_Counts'].apply(lambda x: sorted(x, key=x.get, reverse=True)[0])
    cdl_df['Second_Crop_code'] = cdl_df['Crop_Counts'].apply(lambda x: sorted(x, key=x.get, reverse=True)[1] if len(x) > 1 else np.nan)
    cdl_df['Third_Crop_code']  = cdl_df['Crop_Counts'].apply(lambda x: sorted(x, key=x.get, reverse=True)[2] if len(x) > 2 else np.nan)

    cdl_df['Top_crop']    = cdl_df['Top_Crop_code'].map(CDL_CROP_CODES)
    cdl_df['Second_crop'] = cdl_df['Second_Crop_code'].map(CDL_CROP_CODES)
    cdl_df['Third_crop']  = cdl_df['Third_Crop_code'].map(CDL_CROP_CODES)

    cdl_df['Top_crop_share']    = cdl_df.apply(lambda row: row['Crop_shares'].get(row['Top_Crop_code'], np.nan), axis=1)
    cdl_df['Second_crop_share'] = cdl_df.apply(lambda row: row['Crop_shares'].get(row['Second_Crop_code'], np.nan), axis=1)
    cdl_df['Third_crop_share']  = cdl_df.apply(lambda row: row['Crop_shares'].get(row['Third_Crop_code'], np.nan), axis=1)

    cdl_df['Top_Kc']    = cdl_df['Top_Crop_code'].map(CDL_KC_MID)
    cdl_df['Second_Kc'] = cdl_df['Second_Crop_code'].map(CDL_KC_MID)
    cdl_df['Third_Kc']  = cdl_df['Third_Crop_code'].map(CDL_KC_MID)

    total = cdl_df['Top_crop_share'] + cdl_df['Second_crop_share'] + cdl_df['Third_crop_share']
    
    cdl_df['Weighted_Kc'] = (cdl_df['Top_crop_share'] * cdl_df['Top_Kc']
                              + cdl_df['Second_crop_share'] * cdl_df['Second_Kc']
                              + cdl_df['Third_crop_share'] * cdl_df['Third_Kc']) / total

    cdl_cols = ['HUC8', 'Year', 'Top_Crop_code', 'Top_crop', 'Second_Crop_code', 'Second_crop',
                'Third_Crop_code', 'Third_crop', 'Top_crop_share', 'Second_crop_share',
                'Third_crop_share', 'Top_Kc', 'Second_Kc', 'Third_Kc', 'Weighted_Kc']
    
    panel_df = panel_df.merge(cdl_df[cdl_cols], on=['HUC8', 'Year'], how='left')

    # ── Block 2: PET_P correlation (all years, irrigated pixels only) ─────────────────────────
    logger.info('Block 2: extracting PET_P correlation for irrigated pixels...')

    with rio.open(pet_p_corr_raster) as src:
        pet_p_arr = src.read(1).astype(float)
        ras_crs = src.crs      
        
        if src.nodata is not None:
            pet_p_arr[pet_p_arr == src.nodata] = np.nan
        pet_p_affine = src.transform

    pet_p_records = []
    huc8 = huc8.to_crs(ras_crs)  # reproject HUC8 to raster CRS if needed

    for year in years:
        irr_file = irrigated_cropland_dir / f'Irrigated_cropland_{year}.tif'
        irr_arr = read_raster_arr_object(irr_file, get_file=False).astype(float)
        irr_arr[irr_arr == no_data_value] = np.nan

        # keep PET_P only for irrigated pixels (value == 1)
        masked_pet_p = np.where(irr_arr == 1, pet_p_arr, no_data_value)

        stats = zonal_stats(huc8, masked_pet_p, affine=pet_p_affine, stats=['mean'], nodata=no_data_value)

        for feat, stat in zip(huc8.itertuples(), stats):
            pet_p_records.append({'HUC8': feat.HUC8, 'Year': year, 'mean_pet_p_corr': stat['mean']})

    panel_df = panel_df.merge(pd.DataFrame(pet_p_records), on=['HUC8', 'Year'], how='left')

    # ── Block 3: WTD (all years, irrigated pixels only) ───────────────────────────────────────
    logger.info('Block 3: extracting WTD for irrigated pixels...')

    with rio.open(wtd_raster) as src:
        wtd_arr = src.read(1).astype(float)
        ras_crs = src.crs 
        
        if src.nodata is not None:
            wtd_arr[wtd_arr == src.nodata] = np.nan
        wtd_affine = src.transform

    wtd_records = []
    huc8 = huc8.to_crs(ras_crs)  # reproject HUC8 to raster CRS if needed

    for year in years:
        irr_file = irrigated_cropland_dir / f'Irrigated_cropland_{year}.tif'
        irr_arr = read_raster_arr_object(irr_file, get_file=False).astype(float)
        irr_arr[irr_arr == no_data_value] = np.nan

        # keep WTD only for irrigated pixels (value == 1)
        masked_wtd = np.where(irr_arr == 1, wtd_arr, no_data_value)

        stats = zonal_stats(huc8, masked_wtd, affine=wtd_affine, stats=['mean'], nodata=no_data_value)

        for feat, stat in zip(huc8.itertuples(), stats):
            wtd_records.append({'HUC8': feat.HUC8, 'Year': year, 'Mean_WTD': stat['mean']})

    panel_df = panel_df.merge(pd.DataFrame(wtd_records), on=['HUC8', 'Year'], how='left')

    # ── Block 4: Water source (from AQ_Region lookup) ─────────────────────────────────────────
    logger.info('Block 4: adding water source from AQ_Region...')

    panel_df['Water_source'] = panel_df['AQ_Region'].map(aq_region_water_source)

    # ── Block 5: ET_frac from GS_Kc rasters (all years, irrigated pixels only) ───────────────
    logger.info('Block 5: extracting GS_Kc (ET_frac) for irrigated pixels...')

    gs_kc_dir = Path(gs_kc_dir)
    et_frac_records = []

    for year in years:
        kc_file = gs_kc_dir / f'GS_Kc_{year}.tif'

        if not kc_file.exists():
            for feat in huc8.itertuples():
                et_frac_records.append({'HUC8': feat.HUC8, 'Year': year, 'ET_frac': np.nan})
            continue

        with rio.open(kc_file) as src:
            kc_arr = src.read(1).astype(float)
            ras_crs = src.crs 
            
            if src.nodata is not None:
                kc_arr[kc_arr == src.nodata] = np.nan
            kc_affine = src.transform

        irr_file = irrigated_cropland_dir / f'Irrigated_cropland_{year}.tif'
        irr_arr = read_raster_arr_object(irr_file, get_file=False).astype(float)
        irr_arr[irr_arr == no_data_value] = np.nan

        masked_kc = np.where(irr_arr == 1, kc_arr, no_data_value)

        
        huc8 = huc8.to_crs(ras_crs)  # reproject HUC8 to raster CRS if needed
        stats = zonal_stats(huc8, masked_kc, affine=kc_affine, stats=['mean'], nodata=no_data_value)

        for feat, stat in zip(huc8.itertuples(), stats):
            et_frac_records.append({'HUC8': feat.HUC8, 'Year': year, 'ET_frac': stat['mean']})

    panel_df = panel_df.merge(pd.DataFrame(et_frac_records), on=['HUC8', 'Year'], how='left')

    panel_df.to_csv(output_csv, index=False)
    logger.info(f'HUC8 panel saved to {output_csv}\n')