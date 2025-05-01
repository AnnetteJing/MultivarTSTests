import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import f


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
