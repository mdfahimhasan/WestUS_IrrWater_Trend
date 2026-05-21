# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import logging
import sys
import numpy as np
import pandas as pd
import pymannkendall as mk
from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Project root directory (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)


def calculate_rmse( Y_obsv, Y_pred):
    """
    Calculates RMSE value of model prediction vs observed data.

    :param Y_obsv: observed array or panda series object.
    :param Y_pred: prediction array or panda series object.\

    :return: RMSE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    mse_val = mean_squared_error(y_true=Y_obsv, y_pred=Y_pred)
    rmse_val = np.sqrt(mse_val)

    return rmse_val


def calculate_mae( Y_obsv, Y_pred):
    """
    Calculates MAE value of model prediction vs observed data.

    :param Y_obsv: observed array or panda series object.
    :param Y_pred: prediction array or panda series object.

    :return: MAE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    mae_val = mean_absolute_error(y_true=Y_obsv, y_pred=Y_pred)

    return mae_val


def calculate_r2(Y_obsv, Y_pred):
    """
    Calculates R2 value of model prediction vs observed data.

    :param Y_obsv: observed array or panda series object.
    :param Y_pred: prediction array or panda series object.

    :return: R2 value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    r2_val = r2_score(Y_obsv, Y_pred)

    return r2_val


def calculate_metrics(targets, predictions):
    """
    Calculates regression metrics: RMSE, MAE, R², Normalized RMSE, and Normalized MAE.

    :param targets: array-like or list. True target values.
    :param predictions: array-like or list. Predicted values.

    :return: dict. Dictionary containing:
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'R2': Coefficient of Determination
        - 'Normalized RMSE': RMSE divided by the mean of targets
        - 'Normalized MAE': MAE divided by the mean of targets
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
        targets = np.array(targets)

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    ss_res = np.sum((predictions - targets) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    target_mean = np.mean(targets)
    normalized_rmse = np.nan if target_mean == 0 else rmse / target_mean
    normalized_mae  = np.nan if target_mean == 0 else mae  / target_mean

    return {'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Normalized RMSE': normalized_rmse,
            'Normalized MAE': normalized_mae}


def calc_outlier_ranges_IQR(data, axis=None, decrease_lower_range_by=None, increase_upper_range_by=None):
    """
    calculate lower and upper range of outlier detection using IQR method.

    :param data: An array or list. Flattened array or list is preferred. If not flattened, adjust axis argument or
                 preprocess data before giving ito this function.
    :param axis: Axis or axes along which the percentiles are computed. Default set to None for flattened array or list.
    :param decrease_lower_range_by: A user-defined value to decrease lower range of outlier detection.
                                    Default set to None.
    :param increase_upper_range_by: A user-defined value to increase upper range of outlier detection.
                                    Default set to None.

    :return: lower_range, upper_range values of outlier detection.
    """
    q1 = np.nanpercentile(data, 25, axis=axis)
    median = np.nanpercentile(data, 50, axis=axis)
    q3 = np.nanpercentile(data, 75, axis=axis)

    iqr = q3 - q1

    lower_range = np.nanmin([i for i in data if i >= (q1 - 1.5 * iqr)])
    upper_range = np.nanmax([i for i in data if i <= (q3 + 1.5 * iqr)])

    # adjusts lower and upper values by an author-defined range
    if (decrease_lower_range_by is not None) | (increase_upper_range_by is not None):
        if (decrease_lower_range_by is not None) & (increase_upper_range_by is None):
            lower_range = lower_range - decrease_lower_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is None):
            upper_range = upper_range + increase_upper_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is not None):
            lower_range = lower_range - decrease_lower_range_by
            upper_range = upper_range + increase_upper_range_by

    return lower_range, upper_range, median


def calc_outlier_ranges_MAD(data, axis=None, threshold=3, decrease_lower_range_by=None, increase_upper_range_by=None):
    """
    calculate lower and upper range of outlier detection using Median Absolute Deviation (MAD) method.

    A good paper on MAD-based outlier detection:
    https://www.sciencedirect.com/science/article/pii/S0022103113000668

    :param data: An array or list. Flattened array or list is preferred. If not flattened, adjust axis argument or
                 preprocess data before giving ito this function.
    :param axis: Axis or axes along which the percentiles are computed. Default set to None for flattened array or list.
    :param threshold: Value of threshold to use in MAD method.
    :param decrease_lower_range_by: A user-defined value to decrease lower range of outlier detection.
                                    Default set to None.
    :param increase_upper_range_by: A user-defined value to increase upper range of outlier detection.
                                    Default set to None.

    :return: lower_range, upper_range values of outlier detection.
    """
    # Calculate the median along the specified axis
    median = np.nanmedian(data, axis=axis)

    # Calculate the absolute deviations from the median
    abs_deviation = np.abs(data - median)

    # Calculate the median of the absolute deviations
    MAD = np.nanmedian(abs_deviation, axis=axis)

    lower_range = median - threshold * MAD
    upper_range = median + threshold * MAD

    # adjusts lower and upper values by an author-defined range
    if (decrease_lower_range_by is not None) | (increase_upper_range_by is not None):
        if (decrease_lower_range_by is not None) & (increase_upper_range_by is None):
            lower_range = lower_range - decrease_lower_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is None):
            upper_range = upper_range + increase_upper_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is not None):
            lower_range = lower_range - decrease_lower_range_by
            upper_range = upper_range + increase_upper_range_by

    return lower_range, upper_range, median


def empirical_cdf(data):
    """Returns the empirical cumulative distribution function (ECDF) of the data, ignoring NaNs."""

    # Flatten the data
    flatten_arr = data.flatten()

    # Track the non-Nan and NaN indices
    nan_mask = np.isnan(flatten_arr)
    non_nan_indices = np.where(~nan_mask)[0]  # indices of non-Nan values

    # non-NaN values from the flattened array's
    flat_non_nans = flatten_arr[non_nan_indices]

    # Sort the non-NaN values and get the sorting order (indices)
    sorted_non_nan_indices = np.argsort(flat_non_nans)

    # Sort non-NaN values and their original indices
    sorted_flat_non_nans = flat_non_nans[sorted_non_nan_indices]
    sorted_pred_non_nan_indices = np.array(non_nan_indices)[sorted_non_nan_indices]

    # Calculate ECDF for sorted non-NaN values
    n = len(sorted_flat_non_nans)
    ecdf = np.arange(1, n + 1) / n

    # Return sorted non-NaN values, ECDF, and the original non-NaN indices in sorted order
    return sorted_flat_non_nans, ecdf, non_nan_indices[sorted_non_nan_indices], nan_mask


def quantile_mapping(predictions, observed_train):
    """
    Applies quantile mapping by adjusting predictions to follow the distribution of observed_train data,
    and rearranges them back to the original order, including NaN values in their original positions.
    """
    # Step 1: Get ECDF for predictions (sorted and tracked) and the NaN mask
    sorted_pred, pred_quantiles, sorted_pred_non_nan_indices, nan_mask = empirical_cdf(predictions)

    # Step 2: Get ECDF for observed training data (sorted without tracking indices)
    sorted_obs, obs_quantiles, _, _ = empirical_cdf(observed_train)

    # Step 3: Perform quantile mapping (interpolate predicted quantiles into observed distribution)
    corrected_sorted_predictions = np.interp(pred_quantiles, obs_quantiles, sorted_obs)

    # Step 4: Prepare the corrected predictions array (same shape as original flattened array)
    corrected_predictions = np.empty_like(predictions.flatten())

    # Fill in the corrected non-NaN values in the original positions
    corrected_predictions[sorted_pred_non_nan_indices] = corrected_sorted_predictions

    # Step 5: Re-insert NaN values into the corrected predictions
    corrected_predictions[nan_mask] = np.nan

    # Step 6: Reshape the corrected predictions back to the original shape of predictions
    corrected_predictions = corrected_predictions.reshape(predictions.shape)

    return corrected_predictions


def calc_stdv(data):
    data = np.asarray(data)
    stdv = np.std(data, ddof=1)  # Sample standard deviation (ddof=1)
    
    return stdv

def calc_cv(data):
    data = np.asarray(data)
    mean = np.mean(data)
    stdv = np.std(data, ddof=1)  # Sample standard deviation (ddof=1)
    
    cv = stdv / mean if mean != 0 else np.nan  # Avoid division by zero
    
    return cv


def calc_stdv_cv_by_group(df, value_col, group_col):
    """
    Calculate standard deviation and coefficient of variation for each group in a DataFrame.

    Parameters
    ----------
    df : DataFrame with columns [group_col, value_col].
    value_col : Column to calculate stdv and cv for (e.g. 'IWU_Tmean').
    group_col : Column defining groups (e.g. 'cluster').

    Returns
    -------
    DataFrame with one row per group and columns:
        [group_col, stdv]
    """
    records = []
    
    for group, grp in df.groupby(group_col):
        values = grp[value_col].dropna().values
        
        if len(values) < 2:
            logger.warning(f'Group {group}: only {len(values)} valid values — skipping stdv and cv calculation.')
            continue
        
        stdv = calc_stdv(values)
        cv = calc_cv(values)
        
        records.append({group_col: group, 'stdv': stdv, 'cv': cv})

    cols = [group_col, 'stdv', 'cv']
    
    return pd.DataFrame(records)[cols]
    
    
def mann_kendall_trend(values, alpha=0.05, autocorr_correction=True):
    """
    Run Modified Mann-Kendall trend test + Sen's slope on a 1-D array of annual values.

    Uses the Hamed & Rao (1998) modification by default, which corrects for
    autocorrelation common in climate-driven variables. Falls back to the
    original MK test if autocorr_correction=False.

    Parameters
    ----------
    values : array-like
        Annual time-series values (no NaNs — drop them before calling).
    alpha : float
        Significance threshold. Default 0.05.
    autocorr_correction : bool
        If True, use modified MK (Hamed & Rao). If False, use original MK.

    Returns
    -------
    dict with keys:
        trend       : 'increasing', 'decreasing', or 'no trend'
        S_slope     : Sen's slope (units/year)
        intercept   : Sen's intercept
        p_value     : p-value from MK test
        significant : bool, True if p_value < alpha
        tau         : Kendall's tau statistic
    """
    values = np.asarray(values, dtype=float)

    if autocorr_correction:
        result = mk.hamed_rao_modification_test(values, alpha=alpha)
    else:
        result = mk.original_test(values, alpha=alpha)

    return {
        'trend':       result.trend,
        'S_slope':       result.slope,
        'intercept':   result.intercept,
        'p_value':     result.p,
        'significant': result.p < alpha,
        'tau':         result.Tau,
    }

# def add_sen_slope_ci(df, value_col, group_col, year_col='year'):
#     """
    
#     A custom function to calculate Sen's slope confidence intervals (2.5th and 97.5th percentiles).

#     """
#     records = []
    
#     for group, sub_df in df.groupby(group_col):
#         sub_df_sorted = sub_df.sort_values(year_col)
#         x = sub_df_sorted[value_col].dropna().values
        
#         if len(x) < 4:
#             logger.warning(f'Group {group}: only {len(x)} valid values — skipping MK test.')
#             continue
        
#         # custom Sen's slope function to compute CI
#         slope = []
        
#         for i in range(len(x)):
#             for j in range(i + 1, len(x)):        
#                 slope_ij = (x[j] - x[i]) / (j - i)
#                 slope.append(slope_ij)
                
#         slopes = np.sort(slope) # ascending order
        
#         lower_ci = np.percentile(slopes, 2.5)  # 2.5th percentile slope
#         upper_ci = np.percentile(slopes, 97.5)  # 97.5th percentile slope
        
#         records.append({
#             group_col: group,       
#             'S_slope_lower_ci': lower_ci,
#             'S_slope_upper_ci': upper_ci
#         })

#     return pd.DataFrame(records)
    
    
def mann_kendall_trend_by_group(df, value_col, group_col, year_col='year',
                                alpha=0.05, autocorr_correction=True):
    """
    Apply mann_kendall_trend() to each group (e.g. cluster) in a DataFrame.

    Parameters
    ----------
    df               : DataFrame with columns [group_col, year_col, value_col].
    value_col        : Column to test for trend (e.g. 'IWU_Tmean').
    group_col        : Column defining groups (e.g. 'cluster').
    year_col         : Year column for sorting. Default 'year'.
    alpha            : Significance threshold. Default 0.05.
    autocorr_correction : Passed to mann_kendall_trend(). Default True.

    Returns
    -------
    DataFrame with one row per group and columns:
        [group_col, trend, S_slope, intercept, p_value, significant, tau]

    Example
    -------
    results = mann_kendall_trend_by_group(pdf_annual_clustered,
                                          value_col='IWU_Tmean',
                                          group_col='cluster')
    """
    records = []
    
    for group, grp in df.groupby(group_col):
        grp_sorted = grp.sort_values(year_col)
        valid = grp_sorted[value_col].dropna().values
        
        if len(valid) < 4:
            logger.warning(f'Group {group}: only {len(valid)} valid values — skipping MK test.')
            continue
        
        result = mann_kendall_trend(valid, alpha=alpha,
                                    autocorr_correction=autocorr_correction)
        result[group_col] = group
        records.append(result)
    
    # Merge the results
    results_df = pd.DataFrame(records)

    cols = [group_col, 'trend', 'S_slope', 'intercept', 'p_value', 'significant', 'tau']

    return results_df[cols]