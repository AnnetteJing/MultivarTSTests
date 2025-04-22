import numpy as np
from collections import defaultdict
from scipy.stats import norm
from statsmodels.distributions.copula.api import GaussianCopula
from typing import Optional

BURNIN = 100  # b
D = 2
TREND_FUNCS = [np.sqrt, np.sin]
P = 2
L = {
    0: np.array(
        [
            [1.0, 0.0],
            [0.6, 1.0],
        ]
    ),
    1: np.array(
        [
            [1.2, 0.0],
            [0.2, 0.8],
        ]
    ),
}
INIT_BETA = 0.4
SHRINKAGE = 0.6
ARCH_PARAM_SUM = 0.5


def gaussian_ar_dgp(
    num_timesteps: int,
    window: int,
    cyclo_stationary: bool = True,
    mean_factor: Optional[float] = None,
    cov_factor: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, list]]:
    total_len = window + BURNIN + num_timesteps
    wn = np.random.multivariate_normal(
        mean=np.zeros(D), cov=np.eye(D), size=total_len
    )  # [w + b + N, D]
    # Simulate residuals & save its covariances
    eps = np.zeros(wn.shape)  # [w + b + N, D]
    eps[:window] = wn[:window]
    covs = []
    # Generate time varying volatility process
    for t in range(window, total_len):
        # ARCH parameters sum to 1 (non-stationary)
        lambda_t = np.mean(eps[t - window : t] ** 2, axis=0)
        if num_timesteps > 500:
            lambda_t = np.minimum(lambda_t, 2)
        # ARCH parameters sum to ARCH_PARAM_SUM < 1 (stationary)
        if cyclo_stationary:
            lambda_t *= ARCH_PARAM_SUM
        if t >= window + BURNIN:
            covs.append(np.inner(lambda_t * L[t % P], L[t % P]))
        if np.any(lambda_t < 0) or np.any(np.isnan(lambda_t)):
            raise ValueError(f"{lambda_t = },\n{eps[t - window : t] = }")
        eps[t] = np.sqrt(lambda_t) * wn[t]  # eps_t ~ N(0, diag(lambda_t))
    # Introduce periodic correlation structure
    # eps_t ~ N(0, Lq @ diag(lambda_t) @ Lq.T), q = t mod P
    for q in range(P):
        eps[q::P] = eps[q::P] @ L[q].T
    covs = np.array(covs)  # [N, D, D]
    # Generate Y_bar for the first w steps based on INIT_BETA
    Y_bar_init = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D))
    Y_bar = np.zeros((window + BURNIN + num_timesteps, D))
    Y_bar[0] = INIT_BETA * Y_bar_init + eps[0]
    for t in range(1, window + 1):
        Y_bar[t] = INIT_BETA * Y_bar[t - 1] + eps[t]
    # Simulate the de-meaned target process & save its means
    means = [INIT_BETA * Y_bar[window - 1]]
    for t in range(window, total_len - 1):
        lead = Y_bar[t - window + 1 : t]  # t - w + 1 ~ t - 1
        lag = Y_bar[t - window : t - 1]  # t - w ~ t - 2
        XtY = np.einsum("ti,tj->ij", lead, lag)  # [D, D]
        XtX = np.einsum("ti,tj->ij", lag, lag)  # [D, D]
        beta_t_ols = XtY @ np.linalg.pinv(XtX + 1e-6 * np.eye(D))  # [D, D]
        beta_t = (1 - SHRINKAGE) * beta_t_ols + SHRINKAGE * INIT_BETA * np.eye(D)
        if np.any(np.isnan(beta_t)):
            raise ValueError(
                f"NaNs in beta_t at t = {t},\n {beta_t = },"
                f"{XtY = }, \n{XtX = }, \n{covs[t - window - BURNIN] = }"
            )
        if t >= window + BURNIN:
            means.append(beta_t @ Y_bar[t])
        Y_bar[t + 1] = means[-1] + eps[t + 1]
    means = np.array(means)  # [N, D]
    # Add deterministic trend
    mu = np.stack([f(range(num_timesteps)) for f in TREND_FUNCS]).T  # [N, D]
    targets = Y_bar[window + BURNIN :] + mu  # [N, D]
    means = means + mu  # [N, D]
    # Modify means and covs to be different from DGP for power analysis
    if mean_factor is not None:
        means = mean_factor * means  # [N, D]
    if cov_factor is not None:
        off_diag_mask = np.broadcast_to(~np.eye(D, dtype=bool), (num_timesteps, D, D))
        covs[off_diag_mask] *= cov_factor  # [N, D, D]
    # Save distributions based on means and covs
    distributions = defaultdict(list)
    for t in range(num_timesteps):
        std_t = np.sqrt(np.diag(covs[t]))  # Standard deviations
        corr_t = covs[t] / np.outer(std_t, std_t)  # Correlations
        distributions["marginal"].append(norm(loc=means[t], scale=std_t))
        distributions["copula"].append(GaussianCopula(corr=corr_t, k_dim=D))
    return targets, distributions
