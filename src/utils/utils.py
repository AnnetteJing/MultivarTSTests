import numpy as np


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
