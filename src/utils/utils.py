import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import inv
from scipy.stats import f
from typing import Sequence
import warnings


def matrix_inverse(mat: np.ndarray) -> np.ndarray:
    """
    Args:
        mat ([D, D]): A square matrix to be inverted
    Returns:
        mat_inv ([D, D]): Inverse of mat. Uses Moore-Penrose pseudo-inverse
            if an exact inverse cannot be found.
    """
    try:
        mat_inv = inv(mat)  # [D, D]
    except (np.linalg.LinAlgError, ValueError):
        warnings.warn(
            "Matrix inversion failed. Falling back to pseudo-inverse.",
            RuntimeWarning,
        )
        mat_inv = np.linalg.pinv(mat)  # [D, D]
    return mat_inv


def hotelling_t2_ppf(q: ArrayLike, dim_rv: int, samp_size: int) -> ArrayLike:
    """
    Args:
        q: Lower tail probabilities
        dim_rv: T2 parameter corresponding to the dimension of the RVs
        samp_size: T2 parameter corresponding to the sample size
    Returns:
        t2_quantiles: Hotelling T2 quantiles corresponding to the lower tail probability q
    """
    dfd = samp_size - dim_rv
    f_quantile = f.ppf(q=q, dfn=dim_rv, dfd=dfd)
    t2_quantile = f_quantile * dim_rv * (samp_size - 1) / dfd
    return t2_quantile


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


def get_unif_grid(num_points: int, num_dims: int) -> np.ndarray:
    """
    Constructs a uniform grid over [0, 1]^D

    Args:
        num_points (P): Number of points in the grid
        num_dims (D): Number of dimensions

    Returns:
        grid ([Z, D]): Z = P^D vectors in [0, 1]^D representing a grid where
        each dimension has points 1/P, 2/P, ..., 1 - 1/P, 1.
    """
    grid_size = 1 / num_points
    grid_1d = np.linspace(grid_size, 1, num_points)  # [P,]
    mesh = np.meshgrid(*tuple([grid_1d] * num_dims), indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, num_dims)
    return grid
