from collections.abc import Callable
import numpy as np
import pathos.multiprocessing as mp
from scipy.stats import rv_continuous

from ..kendall_test import KendallTest


def _simulate_size_single(
    alpha: float,
    num_timesteps: int,
    data_generation_process: Callable[[int], tuple[np.ndarray, ...]], 
    get_dist_from_params: Callable[..., dict[str, list[rv_continuous]]],
) -> list[bool]:
    targets, *params = data_generation_process(num_timesteps=num_timesteps)
    dist = get_dist_from_params(*params)
    kendall_test = KendallTest(
        targets=targets, 
        joint_dists=dist["joint"], 
        marginal_dists=dist["marginal"], 
        copulas=dist["copula"], 
        verbose=False
    )
    ks_pval, cvm_pval = kendall_test.get_p_values()
    return [ks_pval <= alpha, cvm_pval <= alpha]


def simulate_size(
    alpha: float,
    num_timesteps: int,
    data_generation_process: Callable[[int], tuple[np.ndarray, ...]], 
    get_dist_from_params: Callable[..., dict[str, list[rv_continuous]]],
    num_repeats: int = 500,
) -> np.ndarray:
    """
    alpha: Nominal size in (0, 1). Rejects if p-value <= alpha
    num_timesteps (N): Number of timesteps to simulate the time series. 
        Passed in as an argument of data_generation_process
    data_generation_process: Function that takes in `num_timesteps` and returns
        - targets: [N, D] array of realized forecasting targets denoted by 
            Y_{t + h}, t = W, ..., T - h = N + W - 1, in the paper
        - params: An arbitrary tuple that defines the forecast distributions denoted by
            Hat{F}_t, t = W, ..., T - h = N + W - 1, in the paper
    get_dist_from_params: Function that takes in `data_generation_process`'s `params` output
        and returns a dictionary with the following keys and values
        - joint: Length N list of D-dimensional continuous forecast densities
            Denoted by Hat{F}_t, t = W, ..., T - h = N + W - 1, in the paper
        - marginal: Length N list of marginals corresponding to each Hat{F}_t
            marginal_dists[t].cdf(y) = [Hat{F}_{t, 1}(y_1), ..., Hat{F}_{t, D}(y_D)]
        - copula: Length N list of copulas corresponding to each Hat{F}_t
            copulas[t].cdf(u) = Hat{C}_t(u)
    """
    available_cpus = mp.cpu_count()
    num_workers = 1 if available_cpus == 1 else (available_cpus - 1)
    print(f"Running Monte Carlo size simulation on {num_workers} CPUs...")
    with mp.ProcessingPool(nodes=num_workers) as pool:
        rejects = pool.map(
            lambda _: _simulate_size_single(
                alpha=alpha, 
                num_timesteps=num_timesteps,
                data_generation_process=data_generation_process,
                get_dist_from_params=get_dist_from_params,
            ), 
            range(num_repeats)
        )
    return np.array(rejects)

