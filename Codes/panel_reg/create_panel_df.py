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

from Codes.panel_reg.panel_utils import BuildPanelDF

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------
# Panel DataFrame creation
# -----------------------------------------------------------------------------------------------------------------
monthly_data_dirs = {
            'ET_mm'     : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_cropET/monthly', 'mean'),
            'IWU_v1_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v1_current', 'mean'),
            'IWU_v2_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v2_current_prev1', 'mean'),
            'IWU_v3_mm' : (PROJECT_ROOT / 'Data_main/rasters/IWU/IWU_monthly/peff_v3_current_prev2', 'mean'),
            'Precip_mm' : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Precip/monthly_masked', 'mean'),
            'Tmean_C' : (PROJECT_ROOT / 'Data_main/rasters/PRISM_Tmean/monthly', 'mean')
        }
annual_data_dirs = {
            'Irr_area_ha' : (PROJECT_ROOT / 'Data_main/rasters/Irrigated_area', 'sum')
        }

static_data_dirs = {
    'WTD_Rnd_Frst_m' : (PROJECT_ROOT / 'Data_main/rasters/CONUS_WTD_RF', 'median'),
    'WTD_USGS_m' : (PROJECT_ROOT / 'Data_main/rasters/USGS_Unconfined_WTD', 'median'),
    'Water_source' : (PROJECT_ROOT / 'Data_main/rasters/USGS_GW_%/Water_source_classification/Water_source_classification.tif', 'median')
    }

skip_panel_df_creation = False      #########################
n_workers = 4                       ########### set to number of available cores on server

if __name__ == "__main__":

    panel_df = BuildPanelDF(n_workers=n_workers)
    
    panel_df.create_monthly_panel_dataframe(
        years_list=range(1986, 2023),   # 1986–2023
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
        include_zero_cols=['Precip_mm', 'Water_source'],
        growing_season_months=range(4, 11),
        no_data_value=-9999,
        skip_processing=skip_panel_df_creation)
