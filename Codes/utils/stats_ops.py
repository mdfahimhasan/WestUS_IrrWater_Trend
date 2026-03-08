# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_rmse(Y_pred, Y_obsv):
    """
    Calculates RMSE value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: RMSE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    mse_val = mean_squared_error(y_true=Y_obsv, y_pred=Y_pred)
    rmse_val = np.sqrt(mse_val)

    return rmse_val


def calculate_mae(Y_pred, Y_obsv):
    """
    Calculates MAE value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: MAE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    mae_val = mean_absolute_error(y_true=Y_obsv, y_pred=Y_pred)

    return mae_val


def calculate_r2(Y_pred, Y_obsv):
    """
    Calculates R2 value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: R2 value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = pd.Series(Y_pred)

    r2_val = r2_score(Y_obsv, Y_pred)

    return r2_val


def calculate_metrics(predictions, targets):
    """
    Calculates regression metrics: RMSE, MAE, R², Normalized RMSE, and Normalized MAE.

    :param predictions: array-like or list. Predicted values.
    :param targets: array-like or list. True target values.

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
    r2 = 1 - (np.sum((predictions - targets) ** 2) /
              np.sum((targets - np.mean(targets)) ** 2))

    normalized_rmse = rmse / np.mean(targets)
    normalized_mae = mae / np.mean(targets)

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


def ks_test_pairwise(state_dfs_dict, round_digits=3):
    """
    Performs pairwise Kolmogorov–Smirnov (K–S) tests between multiple state datasets.

    ------------------------------------------------------------------------------------------------------------------
    Kolmogorov–Smirnov (K–S) Test Overview

    source: https://www.geeksforgeeks.org/machine-learning/kolmogorov-smirnov-test-ks-test/
    ------------------------------------------------------------------------------------------------------------------
    The Kolmogorov–Smirnov (K–S) test is a nonparametric statistical test used to compare two empirical distributions.
    It quantifies the maximum absolute difference between their empirical cumulative distribution functions (ECDFs).
    The resulting D-statistic measures how far apart the two distributions are, while the p-value indicates whether
    this difference is statistically significant. Because the K–S test makes no assumptions about the underlying
    distribution shape, it is particularly useful for assessing whether two samples originate from the same
    population. In this function, it is used to evaluate feature distribution similarity across
    states—for example, comparing Utah’s or California’s climatic and irrigation feature spaces
    against the model’s training states.

    Parameters
    ----------
    state_dfs_dict : dict
        Dictionary where keys are state names and values are DataFrames
        containing the column specified by `value_col`.
    round_digits : int, default=3
        Number of decimal places to round D and p-values.

    Returns
    -------
    D_mat : pd.DataFrame
        Matrix of K–S D-statistics.
    p_mat : pd.DataFrame
        Matrix of corresponding p-values.
    """

    state_names = list(state_dfs_dict.keys())
    D_mat = pd.DataFrame(index=state_names, columns=state_names)
    p_mat = pd.DataFrame(index=state_names, columns=state_names)

    for s1 in state_names:
        vals1 = state_dfs_dict[s1].dropna()
        for s2 in state_names:
            vals2 = state_dfs_dict[s2].dropna()

            # The D-value is the maximum vertical distance between the two empirical
            # cumulative distribution functions (ECDFs). It ranges between 0 and 1.
            # 0 → identical distributions; closer to 1 → very different distributions
            D, p = ks_2samp(vals1, vals2)
            D_mat.loc[s1, s2] = round(D, round_digits)

        #  When n1 and n2 are large, even a small shift in distributions produces a tiny p-value.
        # So, with pixel-scale datasets, we’ll almost always reject the null hypothesis of “same distribution.”
        # so, we are ignoring 'p value' for now.
        # p_mat.loc[s1, s2] = round(p, 8)

    return D_mat
