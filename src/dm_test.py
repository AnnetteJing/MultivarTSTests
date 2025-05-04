from git import Optional
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_t

from src.utils.utils import get_autocovariance, matrix_inverse, HotellingT2


class MultivarDMHLN:
    def __init__(
        self,
        ma_lag: int,
        loss_diff: np.ndarray = None,
        errors1: np.ndarray = None,
        errors2: np.ndarray = None,
        num_samples: int = 100000,
    ):
        """
        Arguments:
            ma_lag (q): Number of lags in the MA representation
            loss_diff: [D, T] array of loss differential between the two models
                If None, defaults to errors1 - errors.
                Either `loss_diff` or `errors1, errors2` must be given.
            errors1: [D, T] array of forecast errors from model 1 (benchmark model)
            errors2: [D, T] array of forecast errors from model 2
            num_samples (B): Number of Monte Carlo samples for size determination
        """
        if loss_diff is None:
            assert errors1 is not None and errors2 is not None
            assert errors1.shape[0] == errors2.shape[0], (
                "Mismatch in number of variables"
            )
            assert errors1.shape[1] == errors2.shape[1], (
                "Mismatch in number of timesteps"
            )
            loss_diff = errors1 - errors2  # [D, T]
        self.loss_diff = loss_diff  # [D, T]
        self.num_variables = loss_diff.shape[0]  # D
        self.timesteps = loss_diff.shape[1]  # T
        self.ma_lag = ma_lag  # q
        self.num_samples = num_samples  # B
        # Consistent estimator of limiting covariance
        auto_covs = get_autocovariance(
            self.loss_diff, lags=range(ma_lag + 1)
        )  # [q + 1, D, D]
        self.sigma_hat = auto_covs[0] + np.sum(
            auto_covs[1:] + np.transpose(auto_covs[1:], (0, 2, 1)), axis=0
        )  # [D, D]
        self.sigma_hat_inv = matrix_inverse(self.sigma_hat)  # [D, D]
        self.sigma_hat_inv_sqrt = sqrtm(self.sigma_hat_inv)  # [D, D]
        self.sigma_hat_diag_inv = 1 / np.diagonal(self.sigma_hat)  # [D,]
        # Calculate the test statistics
        self.mean_loss_diff = np.mean(self.loss_diff, axis=1)  # [D,]
        self.hln_adjustment = (
            self.timesteps - 1 - 2 * ma_lag + ma_lag * (ma_lag + 1) / self.timesteps
        ) / self.timesteps
        self.dm_stats = self.get_dm_stats()

    @staticmethod
    def _get_chi_sq_stat(
        obs: np.ndarray,
        one_side: bool,
        weights: Optional[np.ndarray | float] = None,
        axis: int = None,
    ) -> float:
        if one_side:
            obs = np.maximum(obs, 0)
        obs_sq = obs**2
        if weights is not None:
            obs_sq = obs_sq * weights
        chi_sq_stat = np.sum(obs_sq, axis=axis)
        return chi_sq_stat

    def get_dm_stats(self, mean_vec: Optional[np.ndarray] = None) -> dict[str, float]:
        mean_loss_diff = (
            self.mean_loss_diff if mean_vec is None else self.mean_loss_diff + mean_vec
        )  # [D,]
        mean_loss_diff_white = self.sigma_hat_inv_sqrt @ mean_loss_diff  # [D,]
        const_adj = self.hln_adjustment * (self.timesteps - 1)
        dm_stats = {
            "1side": const_adj
            * self._get_chi_sq_stat(
                mean_loss_diff, one_side=True, weights=self.sigma_hat_diag_inv
            ).item(),
            "1side_unit": const_adj
            * self._get_chi_sq_stat(
                mean_loss_diff, one_side=True, weights=1 / self.num_variables
            ).item(),
            "1side_lr": const_adj
            * self._get_chi_sq_stat(mean_loss_diff_white, one_side=True).item(),
            "2side": const_adj
            * self._get_chi_sq_stat(mean_loss_diff_white, one_side=False).item(),
        }
        return dm_stats

    def _monte_carlo_sampling(self, resample: bool = False) -> None:
        if resample or not hasattr(self, "mc_stats"):
            # Generate Monte Carlo samples for size determination
            self.mc_samples = multivariate_t.rvs(
                shape=self.sigma_hat,
                df=self.timesteps - 1,
                size=self.num_samples,
            )  # [B, D]
            self.mc_stats = {
                "1side": self._get_chi_sq_stat(
                    self.mc_samples,
                    one_side=True,
                    weights=self.sigma_hat_diag_inv,
                    axis=1,
                ),
                "1side_unit": self._get_chi_sq_stat(
                    self.mc_samples,
                    one_side=True,
                    weights=1 / self.num_variables,
                    axis=1,
                ),
                "1side_lr": self._get_chi_sq_stat(
                    (self.sigma_hat_inv_sqrt @ self.mc_samples.T).T,
                    one_side=True,
                    axis=1,
                ),
            }

    def test(self) -> None:
        # Monte Carlo sampling for one-sided tests
        self._monte_carlo_sampling()
        # Hotelling T2 distribution for two-sided test
        self.t2_dist = HotellingT2(
            dim_rv=self.num_variables,
            samp_size=self.timesteps - 1,
        )

    def get_rejection_threshold(self, alpha: float = 0.1) -> dict[str, float]:
        rej_thresh = dict()
        for test_type, mc_stat in self.mc_stats.items():
            rej_thresh[test_type] = np.quantile(mc_stat, 1 - alpha).item()
        rej_thresh["2side"] = self.t2_dist.ppf(1 - alpha).item()
        return rej_thresh

    def get_p_value(self) -> dict[str, float]:
        pvals = dict()
        for test_type, mc_stat in self.mc_stats.items():
            pvals[test_type] = np.mean(mc_stat > self.dm_stats[test_type]).item()
        pvals["2side"] = 1 - self.t2_dist.cdf(self.dm_stats["2side"]).item()
        return pvals
