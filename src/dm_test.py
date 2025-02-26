import numpy as np


def get_autocovariance(x: np.ndarray, lag: int):
    """
    x: [D, T] array, with D being the number of variables & T the number of timesteps
    lag: Lag of autocovariance to compute
    ---
    auto_cov: [D, D] array of the empirical autocovariance Cov(X_t, X_{t - lag})
    """
    x_mean = np.mean(x, axis=1, keepdims=True)  # [D, 1]
    x_centered = x - x_mean  # [D, T]


class MultivarDMHLN:
    def __init__(self, errors1: np.ndarray, errors2: np.ndarray):
        """
        errors1: [D, T] array of forecast errors from model 1 (benchmark model)
        errors2: [D, T] array of forecast errors from model 2
        """
        assert errors1.shape[0] == errors2.shape[0], "Mismatch in number of variables"
        assert errors1.shape[1] == errors2.shape[1], "Mismatch in number of timesteps"
        self.loss_diff = errors1 - errors2  # [D, T]
