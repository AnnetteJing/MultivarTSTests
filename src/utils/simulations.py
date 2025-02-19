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

