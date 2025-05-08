import numpy as np
import argparse
from functools import partial

from src.utils.error_dgps import mariano_preve_dgp


def main():
    raise NotImplementedError("Implementation not finished yet")
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, default=1000)
    parser.add_argument("-t", "--timesteps", type=int, default=200)
    parser.add_argument("-d", "--dim", type=int, default=3)
    parser.add_argument("-q", "--ma-lag", type=int, default=3)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument(
        "-e", "--effect-size", type=float, nargs="+", default=[5, 1, 0.5, 0.1]
    )
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--rho", type=float, default=0.5)
    args = parser.parse_args()
    dgp = partial(
        mariano_preve_dgp,
        dim=args.dim,
        ma_lag=args.ma_lag,
        theta=args.theta,
        rho=args.rho,
    )
    # Mean vectors to test (null, sparse, dense, dense with decaying size)
    mean_vecs = {
        "null": None,
        "sparse": np.zeros(args.dim),
        "dense": np.ones(args.dim),
        "dense_split": np.ones(args.dim),
        "dense_decay": 1 / np.array(range(1, args.dim + 1)) ** 2,
    }
    mean_vecs["sparse"][0] = 1.0
    mean_vecs["dense_split"][1::2] *= -1
    # Effect sizes of alternatives
    effect_sizes = np.array(args.effect_size)
    effect_sizes = np.concatenate((effect_sizes, -effect_sizes[::-1]))
