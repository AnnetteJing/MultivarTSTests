import math
import numpy as np
from scipy.signal import convolve2d
from scipy.stats import ecdf, rv_continuous
from typing import Optional
from collections.abc import Sequence
from numpy.typing import ArrayLike
from tqdm.auto import tqdm


class KendallTest:
    def __init__(
        self,
        targets: ArrayLike,
        joint_dists: Sequence[rv_continuous],
        marginal_dists: Sequence[rv_continuous],
        copulas: Sequence[rv_continuous],
        num_sim: int = 500,
        num_bootstrap: int = 1000,
        block_len: Optional[int] = None,
        grid: Optional[ArrayLike] = None,
        verbose: bool = True,
    ):
        """
        targets: [N, D] array of realized forecasting targets
            Denoted by Y_{t + h}, t = W, ..., T - h = N + W - 1, in the paper
        joint_dists: Length N list of D-dimensional continuous forecast densities
            joint_dists[t].cdf(y) = Hat{F}_t(y), t = W, ..., T - h = N + W - 1
        marginal_dists: Length N list of marginals corresponding to each Hat{F}_t
            marginal_dists[t].cdf(y) = [Hat{F}_{t, 1}(y_1), ..., Hat{F}_{t, D}(y_D)]
        copulas: Length N list of copulas corresponding to each Hat{F}_t
            copulas[t].cdf(u) = Hat{C}_t(u)
        num_sim (I): Number of samples used for simulating the Kendall distribution of Hat{C}_t
        num_bootstrap (B): Number of bootstrap replications
        block_len (l): Length of each block in the block bootstrap
            Defaults to floor(N**(1/2.1))
        grid: [Z,] array that partitions [0, 1]. Defaults to [0, 0.001, 0.002, ..., 0.999, 1]
        verbose: Whether to print progress bars, updates, and warnings
        ---
        V_tilde: [N,] array of partial empirical Kendall variables
            Tilde{V}_t = Hat{C}_t(Tilde{U}_t), where the partial empirical PITs Tilde{U}_t is
            defined by Tilde{U}_{t, d} = Hat{F}_{t, d}(Y_{t + h, d})
        kendall_dists: [N, Z] array representing the input Kendall distributions,
            the t-th row corresponds to kendall_dists[t], i.e. K_{Hat{C}_t}
        V_tilde_indicators: [N, Z] array; (t, z)-th element is 1{V_tilde[t] <= grid[z]}
        V_tilde_dist: [Z,] array representing the empirical distribution of V_tilde
            Denoted by Tilde{K}_N in the paper
        ep: [Z,] array representing the Kendall empirical process
            Denoted by Psi_N in the paper
        ks_stat: Kolmogorov-Smirnov (KS) statistic associated with the Kendall empirical process
        cvm_stat: Cramer von Mises (CvM) statistic associated with the Kendall empirical process
        bootstrap_moving_sums: [N - l + 1, Z] array where each row is the inner sum
            of the bootstrapped process
        """
        self.targets = np.array(targets)  # [N, D]
        self.num_samples = self.targets.shape[0]  # N
        self.num_bootstrap = num_bootstrap  # B
        self.block_len = (
            math.floor(self.num_samples ** (1 / 2.1))
            if block_len is None
            else int(block_len)
        )  # l
        assert self.block_len > 2, "Not enough data for bootstrapping"
        self.grid = np.linspace(0, 1, 1001) if grid is None else np.array(grid)
        # Define self.V_tilde & self.kendall_dists
        self._simulate_kendall(
            joint_dists=joint_dists,
            marginal_dists=marginal_dists,
            copulas=copulas,
            num_sim=num_sim,
            verbose=verbose,
        )
        # Calculate 1{V_tilde[t] <= grid[z]} for each t & z
        self.V_tilde_indicators = self.V_tilde[:, np.newaxis] <= self.grid  # [N, Z]
        # Represent the empirical distribution of V_tilde as an array
        self.V_tilde_dist = np.mean(self.V_tilde_indicators, axis=0)  # [Z,]
        # Define self.ep
        self._evaluate_process()
        # Calculate test statitics
        self.ks_stat = self._ks_statistic(self.ep)
        self.cvm_stat = self._cvm_statistic(self.ep)
        # Calculate the length-l inner sums of the bootstrapped process
        self._prepare_bootstrap()
        # Generate bootstrap distributions of the test statistics
        self.generate_bootstrap_dist()

    def _simulate_kendall(
        self,
        joint_dists: Sequence[rv_continuous],
        marginal_dists: Sequence[rv_continuous],
        copulas: Sequence[rv_continuous],
        num_sim: int,
        verbose: bool,
    ) -> None:
        V_tilde = []
        kendall_dists = []
        for t, Yt in enumerate(tqdm(self.targets, disable=not verbose)):
            # Calculate partially empirical PITs (pePITs)
            pe_pits_t = marginal_dists[t].cdf(Yt)
            # Calculate a sample of fully empirical PITs (fePITs)
            hat_Yt_samp = joint_dists[t].rvs(size=num_sim)  # [I, D]
            fe_pits_t = marginal_dists[t].cdf(hat_Yt_samp)  # [I, D]
            # Copula-transform pePITs
            V_tilde_t = copulas[t].cdf(pe_pits_t).item()
            V_tilde.append(V_tilde_t)
            # Copula-transform fePITs to get empirical Kendall distribution
            kendall_t = ecdf(copulas[t].cdf(fe_pits_t)).cdf
            kendall_dists.append(kendall_t.evaluate(self.grid))
        self.V_tilde = np.array(V_tilde)  # [N,]
        self.kendall_dists = np.array(kendall_dists)  # [N, Z]

    def _evaluate_process(self) -> None:
        mean_kendall_dist = np.mean(self.kendall_dists, axis=0)  # [Z,]
        self.ep = np.sqrt(self.num_samples) * (
            self.V_tilde_dist - mean_kendall_dist
        )  # [Z,]

    @staticmethod
    def _ks_statistic(ep: np.ndarray) -> float:
        return np.max(np.abs(ep)).item()

    @staticmethod
    def _cvm_statistic(ep: np.ndarray) -> float:
        return np.mean(ep**2).item()

    def _prepare_bootstrap(self) -> None:
        bootstrap_summands = self.V_tilde_indicators - self.V_tilde_dist  # [N, Z]
        self.bootstrap_moving_sums = convolve2d(
            bootstrap_summands,  # [N, Z]
            np.ones((self.block_len, 1)),  # Kernel for moving sum, [l, 1]
            mode="valid",
        )  # [N - l + 1, Z]

    def _get_bootstrap_statistics(self) -> tuple[float, float]:
        """
        bootstrap_ks_stat: KS statistic based on the bootstrapped process
            Denoted by kappa_N^{b*} in the paper
        bootstrap_cvm_stat: CvM statistic based on the bootstrapped process
            Denoted by c_N^{b*} in the paper
        """
        # Sample eta_j ~ N(0, 1/l) for j = W, ..., N + W - l
        # Note: np.random.normal's scale is standard deviation so we set it to 1/sqrt(l)
        random_weights = np.random.normal(
            loc=0,
            scale=1 / np.sqrt(self.block_len),
            size=self.bootstrap_moving_sums.shape[0],
        )  # [N - l + 1,]
        # Evaluate the bootstrap empirical process (Psi_N^{b*} in the paper)
        bootstrap_ep = np.sum(
            random_weights[:, np.newaxis]
            * self.bootstrap_moving_sums,  # [N - l + 1, Z]
            axis=0,
        ) / np.sqrt(self.num_samples)  # [Z,]
        # Calculate the associated test statistics
        bootstrap_ks_stat = self._ks_statistic(bootstrap_ep)
        bootstrap_cvm_stat = self._cvm_statistic(bootstrap_ep)
        return bootstrap_ks_stat, bootstrap_cvm_stat

    def generate_bootstrap_dist(self) -> None:
        """
        ks_bootstrap_dist: [B,] array of bootstrapped KS statistics
        cvm_bootstrap_dist: [B,] array of bootstrapped CvM statistics
        """
        ks_bootstrap_dist = []
        cvm_bootstrap_dist = []
        for _ in range(self.num_bootstrap):
            bootstrap_ks_stat, bootstrap_cvm_stat = self._get_bootstrap_statistics()
            ks_bootstrap_dist.append(bootstrap_ks_stat)
            cvm_bootstrap_dist.append(bootstrap_cvm_stat)
        self.ks_bootstrap_dist = np.array(ks_bootstrap_dist)
        self.cvm_bootstrap_dist = np.array(cvm_bootstrap_dist)

    def get_rejection_thresholds(self, alpha: float = 0.1) -> tuple[float, float]:
        """
        ks_alpha: 100*(1 - alpha)% quantile of the bootstrapped KS statistics
        cvm_alpha: 100*(1 - alpha)% quantile of the bootstrapped CvM statistics
        """
        ks_alpha = np.quantile(self.ks_bootstrap_dist, 1 - alpha).item()
        cvm_alpha = np.quantile(self.cvm_bootstrap_dist, 1 - alpha).item()
        return ks_alpha, cvm_alpha

    def get_p_values(self) -> tuple[float, float]:
        """
        ks_pval: p-value of the KS test
        cvm_pval: p-value of the CvM test
        """
        ks_pval = np.mean(self.ks_bootstrap_dist > self.ks_stat).item()
        cvm_pval = np.mean(self.cvm_bootstrap_dist > self.cvm_stat).item()
        return ks_pval, cvm_pval
