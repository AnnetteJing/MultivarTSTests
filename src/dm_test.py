import numpy as np
from typing import Sequence


def get_autocovariance(x: np.ndarray, lags: Sequence[int] | int):
    """
    Arguments:
        x: [D, T] array, with D being the number of variables & T the number of timesteps
            [T,] arrays will be reshaped to [1, T]
        lags: Length L sequence of lags of autocovariance to compute
    Returns:
        auto_covs: [L, D, D] array of the empirical autocovariances
            auto_covs[l] = hat{Cov}(X_t, X_{t - l})
            = Average of outer(x_t - x_mean, x_{t - l} - x_mean) across t = l + 1, ..., T
    """
    if isinstance(lags, int):
        lags = [lags]
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    x_mean = np.mean(x, axis=1, keepdims=True)  # [D, 1]
    x_centered = x - x_mean  # [D, T]
    auto_covs = []
    for lag in lags:
        # Empirical estimate of Cov(X_t, X_{t - lag}) (not de-biased)
        auto_cov_l = np.einsum(
            "it,jt->ijt", np.roll(x_centered, lag)[:, lag:], x_centered[:, lag:]
        ).mean(axis=2)  # [D, D]
        auto_covs.append(auto_cov_l)
    auto_covs = np.array(auto_covs)  # [L, D, D]
    return auto_covs


class MultivarDMHLN:
    def __init__(self, errors1: np.ndarray, errors2: np.ndarray, ma_lag: int):
        """
        Arguments:
            errors1: [D, T] array of forecast errors from model 1 (benchmark model)
            errors2: [D, T] array of forecast errors from model 2
            ma_lag (q): Number of lags in the MA representation
        """
        assert errors1.shape[0] == errors2.shape[0], "Mismatch in number of variables"
        assert errors1.shape[1] == errors2.shape[1], "Mismatch in number of timesteps"
        self.num_variables = errors1.shape[0]  # D
        self.timesteps = errors1.shape[1]  # T
        self.ma_lag = ma_lag  # q
        self.loss_diff = errors1 - errors2  # [D, T]
        auto_covs = get_autocovariance(
            self.loss_diff, lags=range(ma_lag + 1)
        )  # [q + 1, D, D]
        self.sigma_hat = auto_covs[0] + np.sum(
            auto_covs[1:] + np.transpose(auto_covs[1:], (0, 2, 1)), axis=0
        )  # [D, D]
        self.hln_adjustment = (
            self.timesteps - 1 - 2 * ma_lag + ma_lag * (ma_lag + 1) / self.timesteps
        ) / self.timesteps
