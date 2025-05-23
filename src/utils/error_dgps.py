import numpy as np


def mariano_preve_dgp(
    num_timesteps: int, dim: int, ma_lag: int, theta: float, rho: float
) -> np.ndarray:
    """
    Args:
        num_timesteps (T): Number of time steps to generate
        dim (D): Dimension of the target, i.e. number of variables
        ma_lag (q): Number of lags in the moving average process
        theta: Parameter controlling strength of serial correlation
        rho: Parameter controlling strength of contemporaneous correlation
    Returns:
        ma_series ([D, T]): MA(q) series generated following Section 4 of Mariano & Preve (2012)
    """
    # Generate white noise series
    eps_cov = rho * np.ones((dim, dim))
    np.fill_diagonal(eps_cov, 1.0)
    eps = np.random.multivariate_normal(
        mean=np.zeros(dim), cov=eps_cov, size=num_timesteps + ma_lag
    )  # [T + q, D]
    ma_coef = np.kron(
        theta ** np.arange(1, ma_lag + 1), 1 / np.sqrt(np.arange(1, dim + 1))
    ).reshape(-1, dim)  # [q, D]
    ma_coef = np.vstack((np.ones(dim), ma_coef))  # [q + 1, D]
    # Calculate moving average of the white noise series
    eps_windows = np.lib.stride_tricks.sliding_window_view(
        eps, window_shape=ma_lag + 1, axis=0
    )  # [T, D, q + 1]
    ma_series = np.einsum("tdj,jd->dt", eps_windows, ma_coef)  # [D, T]
    return ma_series
