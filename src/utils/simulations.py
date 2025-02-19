from collections.abc import Callable
import numpy as np
import pathos.multiprocessing as mp
from scipy.stats import rv_continuous
from tqdm.auto import tqdm

from ..kendall_test import KendallTest


def _simulate_kendall_single(
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
        verbose=False,
    )
    ks_pval, cvm_pval = kendall_test.get_p_values()
    return [ks_pval <= alpha, cvm_pval <= alpha]


def simulate_kendall(
    num_timesteps: int,
    data_generation_process: Callable[[int], tuple[np.ndarray, ...]],
    get_dist_from_params: Callable[..., dict[str, list[rv_continuous]]],
    num_repeats: int = 500,
    alpha: float = 0.1,
) -> np.ndarray:
    """
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
    num_repeats: Number of Monte Carlo replications for the entire data generation and testing process
    alpha: Nominal size in (0, 1). Rejects if p-value <= alpha
    ---
    rejects: [num_repeats, 2] array of Booleans recording the rejection status of each simulation
        - rejects[:, 0]: Rejections for the Kolmogorov-Smirnov (KS) test
        - rejects[:, 1]: Rejections for the Cramer von Mises (CvM) test
    """
    available_cpus = mp.cpu_count()
    num_workers = 1 if available_cpus == 1 else (available_cpus - 1)
    print(
        f"Running {num_repeats} Monte Carlo size simulations on {num_workers} CPUs..."
    )
    with mp.ProcessingPool(nodes=num_workers) as pool:
        rejects = list(
            tqdm(
                pool.imap(
                    lambda _: _simulate_kendall_single(
                        alpha=alpha,
                        num_timesteps=num_timesteps,
                        data_generation_process=data_generation_process,
                        get_dist_from_params=get_dist_from_params,
                    ),
                    range(num_repeats),
                ),
                total=num_repeats,
                desc="Simulating",
                unit="iter",
            )
        )
    return np.array(rejects)
