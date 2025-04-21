import math
import numpy as np
from scipy.signal import convolve2d
from scipy.stats import rv_continuous
from typing import Optional
from collections.abc import Sequence
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from src.utils.utils import get_unif_grid


class JointDensityTest:
    def __init__(
        self,
        targets: ArrayLike,
        marginal_dists: Sequence[rv_continuous],
        copulas: Sequence[rv_continuous],
        num_bootstrap: int = 1000,
        block_len: Optional[int] = None,
        num_periods: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        targets: [N, D] array of realized forecasting targets
            Denoted by Y_{t + h}, t = w, ..., T - h = N + w - 1, in the paper
        marginal_dists: Length N list of marginals corresponding to each Hat{F}_t
            Let Hat{F}_t(y), t = w, ..., T - h = N + w - 1, be the forecast densities
            conditional on a size w look-back window.
            marginal_dists[t].cdf(y) = [Hat{F}_{t, 1}(y_1), ..., Hat{F}_{t, D}(y_D)]
        copulas: Length N list of copulas corresponding to each Hat{F}_t
            copulas[t].cdf(u) = Hat{C}_t(u)
        num_bootstrap (B): Number of bootstrap replications
        block_len (l): Length of each block in the block bootstrap
            Defaults to round(N**(1/3))
        num_periods (p): Number of periods
            If provided, block_len is round to the nearest multiple
        verbose: Whether to print progress bars, updates, and warnings
        """
        self.targets = np.array(targets)  # [N, D]
        self.num_samples = self.targets.shape[0]  # N
        self.num_variables = self.targets.shape[1]  # D
        assert self.num_variables < 4, (
            "Dimension too high, test not recommended due to inefficiency"
        )
        self.num_bootstrap = num_bootstrap  # B
        self.block_len = (
            round(self.num_samples ** (1 / 3)) if block_len is None else int(block_len)
        )  # l
        if num_periods is not None:
            self.block_len = round(self.block_len / num_periods) * num_periods  # l
            self.num_periods = num_periods  # p
        assert self.block_len > 2, "Not enough data for bootstrapping"
        # Save arguments
        self.marginal_dists = marginal_dists
        self.copulas = copulas
        self.verbose = verbose
        # Define grid over [0, 1]^D
        num_points = 20 if self.num_variables > 2 else 50
        self.grid = get_unif_grid(
            num_points=num_points, num_dims=self.num_variables
        )  # [Z, D]
        # Perform the test
        self.test()

    def _get_ep_summands(self) -> np.ndarray:
        summands = []
        for t, Yt in enumerate(tqdm(self.targets, disable=not self.verbose)):
            # Calculate partially empirical PITs (pePITs)
            pe_pits_t = self.marginal_dists[t].cdf(Yt)  # [D,]
            # Evaluate 1{U_t <= u} for each u in self.grid
            indicators = np.all(pe_pits_t <= self.grid, axis=1).astype(float)  # [Z,]
            # Compute psi_t(u) = 1{U_t <= u} - Hat{C}_t(u) for each u in self.grid
            psi_t = indicators - self.copulas[t].cdf(self.grid)  # [Z,]
            summands.append(psi_t)
        summands = np.array(summands)  # [N, Z]
        return summands

    def _prepare_bootstrap(
        self, summands: np.ndarray, avg_summand: np.ndarray
    ) -> np.ndarray:
        bootstrap_summands = summands - avg_summand  # [N, Z]
        bootstrap_moving_sums = convolve2d(
            bootstrap_summands,  # [N, Z]
            np.ones((self.block_len, 1)),  # Kernel for moving sum, [l, 1]
            mode="valid",
        )  # [N - l + 1, Z]
        return bootstrap_moving_sums

    def _get_bootstrap_ks_stat(self, bootstrap_moving_sums: np.ndarray) -> float:
        # Sample eta_j ~ N(0, 1/l) for j = W, ..., N + W - l
        # Note: np.random.normal's scale is standard deviation so we set it to 1/sqrt(l)
        random_weights = np.random.normal(
            loc=0,
            scale=1 / np.sqrt(self.block_len),
            size=bootstrap_moving_sums.shape[0],
        )  # [N - l + 1,]
        # Evaluate the bootstrap empirical process (Tilde{Psi}_N^*)
        bootstrap_ep = np.sum(
            random_weights[:, np.newaxis] * bootstrap_moving_sums,  # [N - l + 1, Z]
            axis=0,
        ) / np.sqrt(self.num_samples)  # [Z,]
        #
        bootstrap_ks_stat = np.max(np.abs(bootstrap_ep)).item()
        return bootstrap_ks_stat

    def test(self) -> None:
        # Evaluate EP summands Tilde{psi}_t(u), t = w, ..., N + w - 1, u in self.grid
        summands = self._get_ep_summands()  # [N, Z]
        # Average the summands (equal to EP / sqrt{N})
        avg_summand = np.mean(summands, axis=0)  # [Z,]
        # Calculate the KS stat of the EP
        self.ks_stat = math.sqrt(self.num_samples) * np.max(np.abs(avg_summand)).item()
        # Get inner moving block sums of the bootstrap EP
        bootstrap_moving_sums = self._prepare_bootstrap(
            summands=summands, avg_summand=avg_summand
        )  # [N - l + 1, Z]
        # Bootstrap
        ks_bootstrap_dist = []
        for _ in range(self.num_bootstrap):
            bootstrap_ks_stat = self._get_bootstrap_ks_stat(
                bootstrap_moving_sums=bootstrap_moving_sums
            )
            ks_bootstrap_dist.append(bootstrap_ks_stat)
        self.ks_bootstrap_dist = np.array(ks_bootstrap_dist)

    def get_rejection_threshold(self, alpha: float = 0.1) -> float:
        """
        Returns:
            ks_alpha: 100*(1 - alpha)% quantile of the bootstrapped KS statistics
        """
        ks_alpha = np.quantile(self.ks_bootstrap_dist, 1 - alpha).item()
        return ks_alpha

    def get_p_value(self) -> float:
        """
        Returns:
            ks_pval: p-value of the KS test
        """
        ks_pval = np.mean(self.ks_bootstrap_dist > self.ks_stat).item()
        return ks_pval
