import numpy as np
from scipy.linalg import null_space
from scipy.stats import multivariate_t
from typing import Sequence

from src.utils.utils import hotelling_t2_ppf


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
    def __init__(
        self,
        errors1: np.ndarray,
        errors2: np.ndarray,
        ma_lag: int,
        num_samples: int = 100000,
    ):
        """
        Arguments:
            errors1: [D, T] array of forecast errors from model 1 (benchmark model)
            errors2: [D, T] array of forecast errors from model 2
            ma_lag (q): Number of lags in the MA representation
            num_samples (B): Number of Monte Carlo samples for size determination
        """
        assert errors1.shape[0] == errors2.shape[0], "Mismatch in number of variables"
        assert errors1.shape[1] == errors2.shape[1], "Mismatch in number of timesteps"
        self.num_variables = errors1.shape[0]  # D
        self.timesteps = errors1.shape[1]  # T
        self.ma_lag = ma_lag  # q
        self.loss_diff = errors1 - errors2  # [D, T]
        self.mean_loss_diff = np.mean(self.loss_diff, axis=1)  # [D,]
        auto_covs = get_autocovariance(
            self.loss_diff, lags=range(ma_lag + 1)
        )  # [q + 1, D, D]
        self.sigma_hat = auto_covs[0] + np.sum(
            auto_covs[1:] + np.transpose(auto_covs[1:], (0, 2, 1)), axis=0
        )  # [D, D]
        self.sigma_hat_inv = np.linalg.pinv(self.sigma_hat)  # [D, D]
        self.hln_adjustment = (
            self.timesteps - 1 - 2 * ma_lag + ma_lag * (ma_lag + 1) / self.timesteps
        ) / self.timesteps
        # DM statistics
        scaling_const = self.hln_adjustment * (self.timesteps - 1)
        self.dm_stats = {
            "vec": np.sqrt(scaling_const) * self.mean_loss_diff,  # [D,]
            "t2": scaling_const
            * (
                self.mean_loss_diff.T @ self.sigma_hat_inv @ self.mean_loss_diff
            ),  # [1,]
        }
        # Monte Carlo samples for size determination
        self.mc_samples = multivariate_t.rvs(
            shape=self.sigma_hat, df=self.timesteps - 1, size=num_samples
        )  # [B, D]
        self.mc_norms = np.einsum(
            "bi,ij,bj->b", self.mc_samples, self.sigma_hat_inv, self.mc_samples
        )  # [B,]

    def test(self, alpha: float = 0.1, tol: float = 1e-3):
        alpha_1side = alpha + tol * 1e3  # Ensure we always enter the while loop
        alpha_l, alpha_r = alpha, 1.0
        while abs(alpha_1side - alpha) > tol:
            alpha_2side = (alpha_l + alpha_r) / 2
            t2_crit_val = hotelling_t2_ppf(
                q=1 - alpha_2side,
                dim_rv=self.num_variables,
                samp_size=self.timesteps - 1,
            )
            accept_2side_bool = self.mc_norms <= t2_crit_val  # [B,] (A2 are True)
            accept_2side = self.mc_samples[accept_2side_bool]  # [A2, D]
            max_idx = np.argmax(accept_2side, axis=0)  # [D,]
            max_accept = accept_2side[max_idx]  # [D, D]
            max_vals = np.diagonal(max_accept)  # [D,]
            accept_1side_bool = np.any(
                self.mc_samples[~accept_2side_bool] <= max_vals, axis=1
            )  # [B - A2,] (A1 - A2 are True)
            num_accept_1side = np.sum(accept_1side_bool) + len(accept_2side)  # A1
            alpha_1side = num_accept_1side / len(self.mc_norms)  # A1 / B
        # Compute hyperplane
        max_accept = max_accept.T  # [D, D], each col is a D-dim point
        normal_vec = null_space(max_accept[1:] - max_accept[0]).flatten()  # [D,]
        if np.inner(normal_vec, np.ones(self.num_variables) + max_accept[0]) < 0:
            normal_vec = -normal_vec
        # Rejection outcome
        if np.inner(normal_vec, self.dm_stats["vec"] - max_accept[0]) > 0:
            reject = self.dm_stats["t2"] > t2_crit_val
        else:
            reject = np.any(self.dm_stats["vec"] > max_vals)
        return reject.item()
