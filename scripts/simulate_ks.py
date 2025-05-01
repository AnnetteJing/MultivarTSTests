import numpy as np
import argparse
from functools import partial

from src.utils.density_dgps import gaussian_ar_dgp
from src.utils.sim_density import simulate_density_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, default=1000)
    parser.add_argument("-t", "--timesteps", type=int, default=200)
    parser.add_argument("-w", "--window", type=int, default=20)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument(
        "--non-stat", action="store_true", help="Uses non-stationary residuals"
    )
    parser.add_argument("--var-factor", type=float, default=None)
    parser.add_argument("--cov-factor", type=float, default=None)
    args = parser.parse_args()
    dgp = partial(
        gaussian_ar_dgp,
        window=args.window,
        cyclo_stationary=not args.non_stat,
        var_factor=args.var_factor,
        cov_factor=args.cov_factor,
    )
    rejects = simulate_density_test(
        num_timesteps=args.timesteps,
        data_generation_process=dgp,
        num_repeats=args.repeats,
        alpha=args.alpha,
        seed=args.seed,
    )
    prob_rej = np.mean(rejects)
    print(f"KS rejects with probability {prob_rej.item():.4f}")


if __name__ == "__main__":
    main()
