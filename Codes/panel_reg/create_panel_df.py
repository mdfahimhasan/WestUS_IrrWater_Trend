# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import sys
import logging
from pathlib import Path

# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Codes.panel_reg.panel_utils import (BuildPanelDF, 
                                         extract_info_30m_cdl, 
                                         build_huc8_postprocess_df)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------
# Input data directories
# -----------------------------------------------------------------------------------------------------------------
monthly_data_dirs = {
            'ET_mm'     : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly', 'mean'),
            'IWU_v1_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v1_current', 'mean'),
            'IWU_v2_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v2_current_prev1', 'mean'),
            'IWU_v3_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v3_current_prev2', 'mean'),
            'IWU_ACFT'  : (PROJECT_ROOT / 'Data_main/rasters/IWU_ACFT/monthly', 'sum'),
            'Precip_mm' : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked', 'mean'),
            'Tmean_C'   : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Tmean/monthly', 'mean')
        }
annual_data_dirs = {
            'Winter_Precip_mm' : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/winter', 'mean'),
            'Irr_area_ha'      : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_area', 'sum')
        }
static_data_dirs = {
            'WTD_Rnd_Frst_m' : (PROJECT_ROOT / 'Data_main/rasters/CONUS_WTD_RF', 'median'),
            'WTD_USGS_m'     : (PROJECT_ROOT / 'Data_main/rasters/USGS_Unconfined_WTD', 'median'),
        }

# -----------------------------------------------------------------------------------------------------------------
# Processing flags                       ########## set True to skip a step
# -----------------------------------------------------------------------------------------------------------------
skip_panel_df_creation = False
skip_extract_cdl_info  = True
skip_build_huc8_postprocess_df = False
n_workers = 8                            # set to number of available cores on server


if __name__ == "__main__":

    panel_df = BuildPanelDF(n_workers=n_workers)

    # -----------------------------------------------------------------------------------------------------------------
    # Monthly panel DataFrame (1986–2023)
    # -----------------------------------------------------------------------------------------------------------------
    panel_df.create_monthly_panel_dataframe(
        years_list=range(1986, 2024),
        HUC8_shapefile=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8_processed.shp',
        HUC8_name_col='HUC8',
        aquifer_region_col='AQ_Region',
        aquifer_name_col='AQ_NAME',
        state_name_col='State',
        irrigated_cropland_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland',
        monthly_data_dirs=monthly_data_dirs,
        annual_data_dirs=annual_data_dirs,
        static_data_dirs=static_data_dirs,
        streamflow_csv_path=PROJECT_ROOT / 'Data_main/rasters/Dayflow/Merged_Dayflow.csv',
        output_csv_path=PROJECT_ROOT / 'Data_main/panel_data/panel_data_monthly.csv',
        column_rename=None,
        include_zero_cols=['Precip_mm'],
        growing_season_months=range(4, 11),
        no_data_value=-9999,
        skip_processing=skip_panel_df_creation)

    # -----------------------------------------------------------------------------------------------------------------
    # CDL crop info extraction (2008–2023)
    # -----------------------------------------------------------------------------------------------------------------
    extract_info_30m_cdl(
        huc8_shape=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8_processed.shp',
        conus_cdl_dir=PROJECT_ROOT / 'Data_main/rasters/USDA_CDL/raw',
        output_dir=PROJECT_ROOT / 'Data_main/rasters/USDA_CDL',
        skip_processing=skip_extract_cdl_info)

    # -----------------------------------------------------------------------------------------------------------------
    # HUC8 panel: CDL (2008–2023) + PET_P correlation + WTD + weighted Kc + ET_frac
    # -----------------------------------------------------------------------------------------------------------------
    build_huc8_postprocess_df(
        huc8_cdl_csv=PROJECT_ROOT / 'Data_main/rasters/USDA_CDL/HUC8_CDL_crop_counts_shares.csv',
        huc8_shape=PROJECT_ROOT / 'Data_main/ref_shapes/WestUS_HUC8_processed.shp',
        pet_p_corr_raster=PROJECT_ROOT / 'Data_main/rasters/PET_P_correlation/PET_P_corr.tif',
        wtd_raster=PROJECT_ROOT / 'Data_main/rasters/CONUS_WTD_RF/wtd_mean_RF_WestUS.tif',
        irrigated_cropland_dir=PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropland',
        gs_kc_dir=PROJECT_ROOT / 'Data_main/rasters/GS_Kc',
        years=list(range(2008, 2024)),
        output_csv=PROJECT_ROOT / 'Data_main/panel_data/huc8_cdl_pet_p_wtd.csv',
        skip_processing=skip_build_huc8_postprocess_df)
