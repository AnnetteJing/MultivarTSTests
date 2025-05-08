import sys
from collections.abc import Callable
import numpy as np
import pathos.multiprocessing as mp
from tqdm.auto import tqdm
from typing import Optional

from src.dm_test import MultivarDMHLN
from src.density_test import JointDensityTest


def _simulate_dm_test_single(
    alpha: float,
    num_timesteps: int,
    ma_lag: int,
    data_generation_process: Callable[[int], np.ndarray],
    effect_sizes: np.ndarray,
    seed: int,
):
    raise NotImplementedError("Implementation not finished yet")
    np.random.seed(seed)
    loss_diff = data_generation_process(num_timesteps=num_timesteps)
    dm_test = MultivarDMHLN(ma_lag=ma_lag, loss_diff=loss_diff)
    dm_test.test()
    rej_thresholds = dm_test.get_rejection_threshold(alpha=alpha)


def _simulate_density_test_single(
    alpha: float,
    num_timesteps: int,
    data_generation_process: Callable[[int], tuple[np.ndarray, dict[str, list]]],
    seed: int,
) -> list[bool]:
    np.random.seed(seed)
    targets, distributions = data_generation_process(num_timesteps=num_timesteps)
    density_test = JointDensityTest(
        targets=targets,
        marginal_dists=distributions["marginal"],
        copulas=distributions["copula"],
        verbose=False,
    )
    density_test.test()
    ks_pval = density_test.get_p_value()
    return ks_pval < alpha


def simulate_density_test(
    num_timesteps: int,
    data_generation_process: Callable[[int], tuple[np.ndarray, dict[str, list]]],
    num_repeats: int = 500,
    alpha: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Arguments:
        num_timesteps (N): Number of timesteps to simulate the time series.
            Passed in as an argument of data_generation_process
        data_generation_process: Function that takes in `num_timesteps` and returns
            - targets: [N, D] array of realized forecasting targets denoted by
                Y_{t + h}, t = W, ..., T - h = N + W - 1, in the paper
            - distributions: Dictionary with the following keys and values
                - joint: Length N list of D-dimensional continuous forecast densities
                Denoted by Hat{F}_t, t = W, ..., T - h = N + W - 1, in the paper
                - marginal: Length N list of marginals corresponding to each Hat{F}_t
                    marginal_dists[t].cdf(y) = [Hat{F}_{t, 1}(y_1), ..., Hat{F}_{t, D}(y_D)]
                - copula: Length N list of copulas corresponding to each Hat{F}_t
                    copulas[t].cdf(u) = Hat{C}_t(u)
        num_repeats: Number of Monte Carlo replications for the entire data generation and testing process
        alpha: Nominal size in (0, 1). Rejects if p-value <= alpha
        seed: Optional master seed for generating sub-seeds passed to each repetition
    Returns:
        rejects: [num_repeats,] array of Booleans recording the rejection status of each simulation
    """
    available_cpus = mp.cpu_count()
    num_workers = 1 if available_cpus == 1 else (available_cpus - 1)
    if seed is None:
        seed = np.random.randint(1000)
        print(f"No seed given, generated random seed between 0 and 1000: {seed}")
    seed_sequence = np.random.SeedSequence(seed)
    sub_seeds = [s.generate_state(1)[0] for s in seed_sequence.spawn(num_repeats)]
    print(
        f"Running {num_repeats} Monte Carlo size simulations on {num_workers} CPUs..."
    )
    with mp.ProcessingPool(nodes=num_workers) as pool:
        rejects = list(
            tqdm(
                pool.imap(
                    lambda sub_seed: _simulate_density_test_single(
                        alpha=alpha,
                        num_timesteps=num_timesteps,
                        data_generation_process=data_generation_process,
                        seed=sub_seed,
                    ),
                    sub_seeds,
                ),
                total=num_repeats,
                desc="Simulating",
                unit="iter",
                file=sys.stdout,
                smoothing=0,
            )
        )
    return np.array(rejects)
