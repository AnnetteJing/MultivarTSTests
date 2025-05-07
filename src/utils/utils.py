import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.linalg import inv
from scipy.stats import f
from numbers import Number
from matplotlib.colors import Colormap
import seaborn as sns
from typing import Sequence, Optional, Literal
import warnings

sns.set_style("darkgrid")


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


class HotellingT2:
    def __init__(self, dim_rv: int, samp_size: int):
        """
        Args:
            dim_rv: T2 parameter corresponding to the dimension of the RVs
            samp_size: T2 parameter corresponding to the sample size
        """
        # Hotelling T2 parameters
        self.dim_rv = dim_rv
        self.samp_size = samp_size
        # Corresponding F parameters
        self.dfn = dim_rv
        self.dfd = samp_size - dim_rv
        # Scaling factor linking the two distributions
        self.scale = self.dfn * (samp_size - 1) / self.dfd

    def ppf(self, q: ArrayLike) -> ArrayLike:
        """
        Args:
            q: Lower tail probabilities
        Returns:
            t2_quantile: Hotelling T2 quantiles corresponding to lower tail probabilities q
        """
        f_quantile = f.ppf(q=q, dfn=self.dfn, dfd=self.dfd)
        t2_quantile = self.scale * f_quantile
        return t2_quantile

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """
        Args:
            x: Quantiles
        Returns:
            t2_prob: Hotelling T2 lower tail probabilities corresponding to quantiles q
        """
        f_x = x / self.scale
        t2_prob = f.cdf(f_x, dfn=self.dfn, dfd=self.dfd)
        return t2_prob


def pd_display_gradient(
    df: pd.DataFrame,
    cmap: Optional[Colormap] = None,
    axis: Optional[Literal[0, 1]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    precision: Optional[int] = None,
) -> pd.DataFrame:
    if cmap is None:
        cmap = sns.color_palette("coolwarm", as_cmap=True)
    if axis is None:
        gmap = df
        if isinstance(min_val, Number):
            gmap = np.maximum(gmap, min_val)
        if isinstance(max_val, Number):
            gmap = np.minimum(gmap, max_val)
        df_gradient = df.apply(pd.to_numeric).style.background_gradient(
            gmap=gmap, cmap=cmap, axis=axis
        )
    else:
        df_gradient = df.apply(pd.to_numeric).style.background_gradient(
            cmap=cmap, axis=axis
        )
    if precision is not None:
        df_gradient = df_gradient.format(precision=precision)
    return df_gradient
