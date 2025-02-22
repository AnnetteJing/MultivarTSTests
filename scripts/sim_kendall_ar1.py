import numpy as np
import argparse
from collections import defaultdict
from scipy.stats import norm, multivariate_normal
from statsmodels.distributions.copula.api import GaussianCopula

from src.utils.simulations import simulate_kendall


D = 3  # Number of variables
Sigma = np.array(
    [[1.0, 0.1, 0.4], [0.1, 1.0, 0.3], [0.4, 0.3, 1.0]]
)  # Covariance matrix of white noise
Sigma2 = np.array(
    [[2.0, 0.6, -0.1], [0.6, 1.0, 0.2], [-0.1, 0.2, 1.5]]
)  # Alternative covariance matrix of white noise
Sigmas = (Sigma, Sigma2)
phi = 0.1  # Common AR(1) coefficient for temporal dependence


def _generate_targets(
    num_timesteps: int, inital_cov: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    Y_0 = np.random.multivariate_normal(mean=np.zeros(D), cov=inital_cov)  # [D,]
    Y = [Y_0]
    for t in range(num_timesteps):
        Y.append(phi * Y[-1] + noise[t])
    return np.array(Y)


def ar1_static_gaussian_dgp(
    num_timesteps: int,
) -> tuple[np.ndarray, dict[str, list]]:
    eps = np.random.multivariate_normal(
        mean=np.zeros(D), cov=Sigma, size=num_timesteps
    )  # [T, D]
    Y = _generate_targets(
        num_timesteps=num_timesteps, inital_cov=Sigma / (1 - phi**2), noise=eps
    )  # [N + 1, D]
    targets = Y[1:]  # [N, D]
    means = phi * Y[:-1]  # [N, D]
    std = np.sqrt(np.diag(Sigma))  # Standard deviations of each variable
    corr = Sigma / np.outer(std, std)  # Correlation matrix
    distributions = defaultdict(list)
    for mean_t in means:
        distributions["joint"].append(multivariate_normal(mean=mean_t, cov=Sigma))
        distributions["marginal"].append(norm(loc=mean_t, scale=std))
    distributions["copula"] = [GaussianCopula(corr=corr, k_dim=D)] * num_timesteps
    return targets, distributions


def ar1_2period_gaussian_dgp(
    num_timesteps: int,
) -> tuple[np.ndarray, dict[str, list]]:
    eps = np.empty((num_timesteps, D), dtype=np.float64)  # [T, D]
    num_timesteps_2 = num_timesteps // 2  # T2
    num_timesteps_1 = num_timesteps - num_timesteps_2  # T1
    eps[0::2] = np.random.multivariate_normal(
        mean=np.zeros(D), cov=Sigma, size=num_timesteps_1
    )  # [T1, D]
    eps[1::2] = np.random.multivariate_normal(
        mean=np.zeros(D), cov=Sigma2, size=num_timesteps_2
    )  # [T2, D]
    Y = _generate_targets(
        num_timesteps=num_timesteps, inital_cov=Sigma / (1 - phi**2), noise=eps
    )  # [N + 1, D]
    targets = Y[1:]  # [N, D]
    means = phi * Y[:-1]  # [N, D]
    # Periods 1 & 2 standard deviations of each variable
    stds = tuple(np.sqrt(np.diag(cov)) for cov in Sigmas)
    # Periods 1 & 2 correlation matrices
    corrs = tuple(cov / np.outer(std, std) for cov, std in zip(Sigmas, stds))
    distributions = defaultdict(list)
    for t, mean_t in enumerate(means):
        period = t % 2
        distributions["joint"].append(
            multivariate_normal(mean=mean_t, cov=Sigmas[period])
        )
        distributions["marginal"].append(norm(loc=mean_t, scale=stds[period]))
    copulas = [GaussianCopula(corr=corr, k_dim=D) for corr in corrs] * num_timesteps_1
    distributions["copula"] = copulas[:num_timesteps]
    return targets, distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, default=500)
    parser.add_argument("-t", "--timesteps", type=int, default=200)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument(
        "--periodic", action="store_true", help="Activates periodic covariance"
    )
    args = parser.parse_args()
    dgp = ar1_2period_gaussian_dgp if args.periodic else ar1_static_gaussian_dgp
    rejects = simulate_kendall(
        alpha=args.alpha,
        num_timesteps=args.timesteps,
        data_generation_process=dgp,
        num_repeats=args.repeats,
    )
    prob_rej = np.mean(rejects, axis=0)
    print(f"KS rejects with probability {prob_rej[0].item():.3f}")
    print(f"CvM rejects with probability {prob_rej[1].item():.3f}")


if __name__ == "__main__":
    main()
