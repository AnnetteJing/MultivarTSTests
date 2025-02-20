import numpy as np
import argparse

from ..src.utils.simulations import simulate_kendall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, default=500)
    parser.add_argument("-t", "--timesteps", type=int, default=200)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    args = parser.parse_args()
    rejects = simulate_kendall(
        alpha=args.alpha,
        num_timesteps=args.timesteps,
        data_generation_process=None,
        get_dist_from_params=None,
        num_repeats=args.repeats,
    )
    prob_rej = np.mean(rejects, axis=0)
    print(f"KS rejects with probability {prob_rej[0].item():.3f}")
    print(f"CvM rejects with probability {prob_rej[1].item():.3f}")


if __name__ == "__main__":
    main()
