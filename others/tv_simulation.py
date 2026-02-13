
import numpy as np

def eta_tv_threshold(
    k: int,
    alpha: float,
    *,
    mu0: float = 0.0,
    sigma_f: float = 0.4,
    ell: float = 4.0,
    n_mc: int = 50_000,
    seed: int = 0,
    jitter: float = 1e-10,
    return_samples: bool = False,
):
    """
    Calibrate eta(k, alpha) by simulation so that P(TV(mu_1:k) > eta) <= alpha
    under a GP prior with exponential kernel:
        K_ij = sigma_f^2 * exp(-|i-j|/ell).

    TV is the discrete total variation on the first k points:
        TV(mu_1:k) = sum_{t=1}^{k-1} |mu_{t+1} - mu_t|.

    Parameters
    ----------
    k : number of dates/time points
    alpha : tail probability level (e.g., 0.05)
    mu0 : constant mean level
    sigma_f : marginal std dev of GP
    ell : length scale in days
    n_mc : number of Monte Carlo draws
    seed : RNG seed
    jitter : small diagonal bump for numerical stability
    return_samples : if True, also return the simulated TV values

    Returns
    -------
    eta : calibrated threshold
    (optional) tv_vals : array of simulated TV values
    """
    if not (isinstance(k, int) and k >= 2):
        raise ValueError("k must be an integer >= 2.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if n_mc < 1000:
        raise ValueError("n_mc should be reasonably large (e.g. >= 1000).")

    rs = np.random.RandomState(seed)

    # time points 1..k
    t = np.arange(1, k + 1, dtype=float)
    D = np.abs(t[:, None] - t[None, :])

    # exponential kernel covariance
    K = (sigma_f**2) * np.exp(-D / ell)

    # Cholesky for sampling MVN(0, K)
    L = np.linalg.cholesky(K + jitter * np.eye(k))

    # Sample mu paths: mu = mu0 + L z, z ~ N(0, I)
    Z = rs.standard_normal(size=(k, n_mc))         # (k, n_mc)
    MU = mu0 + (L @ Z)                             # (k, n_mc)

    # Discrete total variation for each draw
    tv_vals = np.sum(np.abs(np.diff(MU, axis=0)), axis=0)  # (n_mc,)

    # Choose eta so empirical tail <= alpha (conservative via order statistic)
    tv_sorted = np.sort(tv_vals)
    idx = int(np.ceil((1.0 - alpha) * n_mc)) - 1  # 0-based
    idx = min(max(idx, 0), n_mc - 1)
    eta = tv_sorted[idx]

    return (eta, tv_vals) if return_samples else eta


if __name__ == "__main__":
    
    bias = []
    for t in range(1, 15):
        bias.append(eta_tv_threshold(t, 0.05))

    