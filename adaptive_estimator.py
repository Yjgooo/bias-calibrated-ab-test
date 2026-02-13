import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

from visualization import plot_sample_trajectories, plot_sample_trajectories_aligned_start, plot_last_day_lower_bounds


# ============================================================
# 0) Simulator (same as previous, constant N within experiment)
# ============================================================

def half_t(df: float, scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return np.abs(rng.standard_t(df, size=size)) * scale

def sample_b_mixture_half_t(
    K: int,
    pi: float,
    nu: float,
    sigma_pos: float,
    sigma_neg: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.where(rng.random(K) < pi, 1.0, -1.0)
    scales = np.where(s > 0, sigma_pos, sigma_neg)
    h = half_t(df=nu, scale=scales, size=K, rng=rng)
    b = s * h
    return b, s, h

@dataclass
class SyntheticABConfig:
    K: int = 200
    T: int = 14
    lambda_low: float = 1.0
    lambda_high: float = 30.0
    a_mean: float = 0.0
    a_sd: float = 0.05
    pi: float = 0.7
    nu: float = 3.0
    sigma_pos: float = 0.10
    sigma_neg: float = 0.06
    sigma: float = 1.0
    N_T_low: int = 2000
    N_T_high: int = 20000
    N_C_low: int = 2000
    N_C_high: int = 20000
    seed: int = 0

def generate_synthetic_ab_data(cfg: SyntheticABConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    K, T = cfg.K, cfg.T
    t = np.arange(1, T + 1, dtype=float)

    # lambda_i ~ log-uniform
    loglam = rng.uniform(np.log(cfg.lambda_low), np.log(cfg.lambda_high), size=K)
    lam = np.exp(loglam)

    # a_i ~ Normal
    a = rng.normal(cfg.a_mean, cfg.a_sd, size=K)

    # b_i mixture half-t
    b, s, h = sample_b_mixture_half_t(
        K=K, pi=cfg.pi, nu=cfg.nu, sigma_pos=cfg.sigma_pos, sigma_neg=cfg.sigma_neg, rng=rng
    )

    # constant sample sizes within experiment
    N_T = rng.integers(cfg.N_T_low, cfg.N_T_high + 1, size=K)
    N_C = rng.integers(cfg.N_C_low, cfg.N_C_high + 1, size=K)
    N_eff = 1.0 / (1.0 / N_T + 1.0 / N_C)  # (K,)

    # true trajectory
    decay = np.exp(-(t[None, :] - 1.0) / lam[:, None])   # (K,T)
    d_true = a[:, None] + b[:, None] * decay             # (K,T)

    # observed
    sd_obs = cfg.sigma / np.sqrt(N_eff)                  # (K,)
    d_obs = d_true + rng.normal(0.0, sd_obs[:, None], size=(K, T))

    # Delta via provided closed form
    r = np.exp(-1.0 / lam)
    Delta = np.zeros((K, T - 1), dtype=float)
    for k in range(1, T):
        term = (r ** (T - k)) * ((r ** (k - 1)) - (1.0 - r ** k) / (k * (1.0 - r)))
        Delta[:, k - 1] = b * term

    return {
        "a": a, "b": b, "lambda": lam,
        "N_T": N_T, "N_C": N_C, "N_eff": N_eff,
        "d_true": d_true, "d_obs": d_obs,
        "Delta": Delta,
        "meta": {**cfg.__dict__}
    }


# ============================================================
# 1) Per-experiment MLE: (a_i, b_i, lambda_i)
#    Model: d_t ~ N(a + b exp(-(t-1)/lambda), sigma^2/N_eff)
#    With constant N_eff across t, this is (scaled) least squares.
# ============================================================

def _fit_ab_given_lambda(d: np.ndarray, lam: float) -> Tuple[float, float, float]:
    """
    For fixed lambda, fit a,b by OLS:
      d_t = a + b x_t + eps, x_t = exp(-(t-1)/lambda)
    Returns (a_hat, b_hat, rss)
    """
    T = len(d)
    t = np.arange(T, dtype=float)  # 0..T-1 corresponds to (t-1) in your formula
    x = np.exp(-t / lam)
    X = np.column_stack([np.ones(T), x])  # (T,2)
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)  # (2,)
    resid = d - X @ beta
    rss = float(resid @ resid)
    return float(beta[0]), float(beta[1]), rss

def _golden_section_minimize(f, a, b, iters=60):
    """
    Minimize f on [a,b] using golden section search.
    Assumes f is unimodal-ish on [a,b].
    """
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if fc > fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
    x_min = (a + b) / 2
    return x_min, f(x_min)

def fit_one_experiment_mle(
    d: np.ndarray,
    *,
    lambda_bounds: Tuple[float, float] = (0.2, 200.0),
    n_grid: int = 80,
    refine_iters: int = 60,
) -> Dict[str, float]:
    """
    Profile-MLE for one trajectory:
      - grid over log(lambda)
      - refine with golden-section on log(lambda)
      - for each lambda, a,b have closed form via OLS

    Returns: a_hat, b_hat, lambda_hat, rss
    """
    d = np.asarray(d, float)
    lo, hi = lambda_bounds
    if lo <= 0 or hi <= lo:
        raise ValueError("lambda_bounds must be (positive_lo, hi) with hi>lo")

    # coarse grid in log-space
    grid = np.linspace(np.log(lo), np.log(hi), n_grid)
    rss_vals = []
    for u in grid:
        lam = float(np.exp(u))
        _, _, rss = _fit_ab_given_lambda(d, lam)
        rss_vals.append(rss)
    rss_vals = np.array(rss_vals)
    j = int(np.argmin(rss_vals))
    # bracket for refinement
    j_lo = max(0, j - 1)
    j_hi = min(n_grid - 1, j + 1)
    u_lo, u_hi = float(grid[j_lo]), float(grid[j_hi])

    def obj(u):
        lam = float(np.exp(u))
        _, _, rss = _fit_ab_given_lambda(d, lam)
        return rss

    u_hat, rss_hat = _golden_section_minimize(obj, u_lo, u_hi, iters=refine_iters)
    lam_hat = float(np.exp(u_hat))
    a_hat, b_hat, rss = _fit_ab_given_lambda(d, lam_hat)

    return {"a_hat": a_hat, "b_hat": b_hat, "lambda_hat": lam_hat, "rss": float(rss)}

def fit_all_experiments_mle(
    d_obs: np.ndarray,
    N_eff: np.ndarray,
    *,
    lambda_bounds: Tuple[float, float] = (0.2, 200.0),
    n_grid: int = 80,
    refine_iters: int = 60,
) -> Dict[str, Any]:
    """
    Fit each experiment independently.

    Notes:
    - With constant N_eff across days, it only scales the log-likelihood by N_eff,
      so argmin RSS (hence MLE of a,b,lambda) is unchanged.
    - We still return sigma2_hat_i = RSS_i / T for convenience.
    """
    d_obs = np.asarray(d_obs, float)
    N_eff = np.asarray(N_eff, float)
    K, T = d_obs.shape

    a_hat = np.empty(K)
    b_hat = np.empty(K)
    lam_hat = np.empty(K)
    rss = np.empty(K)
    sigma2_hat = np.empty(K)

    for i in range(K):
        fit = fit_one_experiment_mle(
            d_obs[i],
            lambda_bounds=lambda_bounds,
            n_grid=n_grid,
            refine_iters=refine_iters,
        )
        a_hat[i] = fit["a_hat"]
        b_hat[i] = fit["b_hat"]
        lam_hat[i] = fit["lambda_hat"]
        rss[i] = fit["rss"]
        sigma2_hat[i] = fit["rss"] / T  # not weighted; simple plug-in

    return {
        "a_hat": a_hat,
        "b_hat": b_hat,
        "lambda_hat": lam_hat,
        "rss": rss,
        "sigma2_hat": sigma2_hat,
        "meta": {"lambda_bounds": lambda_bounds, "n_grid": n_grid, "refine_iters": refine_iters},
    }


# ============================================================
# 2) EB hyperparameter estimation for:
#    a ~ Normal(mu_a, sd_a^2)
#    log lambda ~ Normal(mu_lam, sd_lam^2)
#    b = s*h with P(s=+1)=pi, h|s ~ Half-t(nu, scale_s)
#    Here nu is fixed and we MLE the scales scale_+, scale_-.
# ============================================================

def _log_t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """
    log pdf of standard Student-t(df) at x (scale=1, loc=0).
    """
    x = np.asarray(x, float)
    v = df
    # log Gamma((v+1)/2) - log Gamma(v/2) - 0.5 log(v*pi) - (v+1)/2 log(1 + x^2/v)
    return (
        math.lgamma((v + 1) / 2.0)
        - math.lgamma(v / 2.0)
        - 0.5 * (math.log(v) + math.log(math.pi))
        - ((v + 1) / 2.0) * np.log1p((x * x) / v)
    )

def _half_t_negloglik_scale(h: np.ndarray, df: float, scale: float) -> float:
    """
    Negative log-likelihood for h >= 0 under Half-t(df, scale).
    Density: f(h) = 2 * t_pdf(h/scale) / scale, h>=0.
    """
    if scale <= 0:
        return float("inf")
    z = h / scale
    log_pdf = math.log(2.0) - math.log(scale) + _log_t_pdf(z, df)
    return float(-np.sum(log_pdf))

def mle_half_t_scale(h: np.ndarray, df: float, scale_bounds: Tuple[float, float] = (1e-6, 10.0)) -> float:
    """
    MLE for scale in Half-t(df, scale), via golden-section on log(scale).
    """
    h = np.asarray(h, float)
    h = h[h >= 0]
    if h.size == 0:
        return float("nan")

    lo, hi = scale_bounds
    if lo <= 0 or hi <= lo:
        raise ValueError("scale_bounds must be (positive_lo, hi) with hi>lo")

    u_lo, u_hi = math.log(lo), math.log(hi)

    def obj(u):
        s = math.exp(u)
        return _half_t_negloglik_scale(h, df=df, scale=s)

    u_hat, _ = _golden_section_minimize(obj, u_lo, u_hi, iters=80)
    return float(math.exp(u_hat))

def empirical_bayes_hyperparams(
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    lambda_hat: np.ndarray,
    *,
    nu_fixed: float = 3.0,
    scale_bounds: Tuple[float, float] = (1e-6, 10.0),
) -> Dict[str, Any]:
    """
    Fit hyperparameters from fitted per-experiment estimates (plug-in EB).

    Returns:
      a_mean, a_sd
      loglambda_mean, loglambda_sd
      pi_hat
      sigma_pos_hat, sigma_neg_hat  (half-t scales for |b| conditional on sign)
    """
    a_hat = np.asarray(a_hat, float)
    b_hat = np.asarray(b_hat, float)
    lambda_hat = np.asarray(lambda_hat, float)

    # Normal for a
    a_mean = float(np.mean(a_hat))
    a_sd = float(np.std(a_hat, ddof=1))

    # Normal for log lambda
    ll = np.log(lambda_hat)
    loglambda_mean = float(np.mean(ll))
    loglambda_sd = float(np.std(ll, ddof=1))

    # mixture sign probability
    pi_hat = float(np.mean(b_hat > 0))

    # Half-t scales for magnitude |b| conditional on sign
    h_pos = np.abs(b_hat[b_hat > 0])
    h_neg = np.abs(b_hat[b_hat < 0])

    # choose broad bounds based on data scale (optional: tighten)
    sigma_pos_hat = mle_half_t_scale(h_pos, df=nu_fixed, scale_bounds=scale_bounds) if h_pos.size > 0 else float("nan")
    sigma_neg_hat = mle_half_t_scale(h_neg, df=nu_fixed, scale_bounds=scale_bounds) if h_neg.size > 0 else float("nan")

    return {
        "a_mean": a_mean, "a_sd": a_sd,
        "loglambda_mean": loglambda_mean, "loglambda_sd": loglambda_sd,
        "pi": pi_hat, "nu": nu_fixed,
        "sigma_pos": sigma_pos_hat, "sigma_neg": sigma_neg_hat,
        "counts": {"K": int(len(a_hat)), "n_pos": int(h_pos.size), "n_neg": int(h_neg.size)},
    }

def delta_k_thresholds(
    T: int,
    alpha: float,
    *,
    # EB prior params for b and lambda
    pi: float,
    nu: float,
    sigma_pos: float,
    sigma_neg: float,
    loglambda_mean: float,
    loglambda_sd: float,
    # MC
    n_mc: int = 100_000,
    seed: int = 0,
) -> Dict[int, float]:
    """
    Return the (alpha) upper-quantile threshold eta_k for Delta_k for k=1..T-1.

    eta_k = quantile_{alpha}(Delta_k) under the EB prior:
      log lambda ~ Normal(loglambda_mean, loglambda_sd^2)
      b = s*h, P(s=+1)=pi, h|s ~ Half-t(nu, sigma_s^2)

    Returns
    -------
    dict mapping k -> eta_k  for k=1..T-1
    """
    if not (isinstance(T, int) and T >= 2):
        raise ValueError("T must be an integer >= 2.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    rng = np.random.default_rng(seed)

    # sample lambda ~ LogNormal
    lam = np.exp(rng.normal(loc=loglambda_mean, scale=loglambda_sd, size=n_mc))
    r = np.exp(-1.0 / lam)  # (n_mc,)

    # sample b = s*h, with h ~ Half-t
    s = np.where(rng.random(n_mc) < pi, 1.0, -1.0)
    scales = np.where(s > 0, sigma_pos, sigma_neg)
    h = np.abs(rng.standard_t(df=nu, size=n_mc)) * scales
    b = s * h  # (n_mc,)

    etas: Dict[int, float] = {}
    for k in range(1, T):  # k=1..T-1
        term = (r ** (T - k)) * ((r ** (k - 1)) - (1.0 - r ** k) / (k * (1.0 - r)))
        Delta = b * term
        etas[k] = float(np.quantile(Delta, alpha))

    return etas


# ============================================================
# 6) NEW: Plot sample trajectories from the simulation (add-on)
#     (No changes to existing code above)
# ============================================================




# ============================================================
# NEW: Simulate *true noiseless* trajectories only (add-on)
#      Uses the same priors as generate_synthetic_ab_data, but
#      does NOT simulate N_T/N_C/N_eff and does NOT add noise.
# ============================================================

def simulate_true_noiseless_trajectories(cfg: SyntheticABConfig) -> Dict[str, Any]:
    """
    Simulate only the latent, noiseless curves:
        d_true_{i,t} = a_i + b_i * exp(-(t-1)/lambda_i)

    using the priors in SyntheticABConfig:
      - log lambda_i ~ Uniform(log(lambda_low), log(lambda_high))
      - a_i ~ Normal(a_mean, a_sd^2)
      - b_i = s_i * h_i, P(s_i=+1)=pi, h_i|s_i ~ Half-t(nu, sigma_s^2)

    Returns
    -------
    dict with:
      a: (K,)
      b: (K,)
      lambda: (K,)
      d_true: (K,T)
      Delta: (K,T-1)  (same closed form as before)
      meta: config snapshot
    """
    rng = np.random.default_rng(cfg.seed)
    K, T = cfg.K, cfg.T
    t = np.arange(1, T + 1, dtype=float)

    # lambda_i ~ log-uniform
    loglam = rng.uniform(np.log(cfg.lambda_low), np.log(cfg.lambda_high), size=K)
    lam = np.exp(loglam)

    # a_i ~ Normal
    a = rng.normal(cfg.a_mean, cfg.a_sd, size=K)

    # b_i mixture half-t
    b, s, h = sample_b_mixture_half_t(
        K=K, pi=cfg.pi, nu=cfg.nu,
        sigma_pos=cfg.sigma_pos, sigma_neg=cfg.sigma_neg,
        rng=rng
    )

    # noiseless trajectory
    decay = np.exp(-(t[None, :] - 1.0) / lam[:, None])   # (K,T)
    d_true = a[:, None] + b[:, None] * decay             # (K,T)

    # Delta via provided closed form (depends only on b, lambda)
    r = np.exp(-1.0 / lam)
    Delta = np.zeros((K, T - 1), dtype=float)
    for k in range(1, T):
        term = (r ** (T - k)) * ((r ** (k - 1)) - (1.0 - r ** k) / (k * (1.0 - r)))
        Delta[:, k - 1] = b * term

    return {
        "a": a,
        "b": b,
        "lambda": lam,
        "sign_s": s,
        "h": h,
        "d_true": d_true,
        "Delta": Delta,
        "meta": {**cfg.__dict__},
    }

def estimate_sigma_hat_from_residuals(
    x: np.ndarray,
    *,
    N_eff: float,
    lambda_bounds: Tuple[float, float] = (0.5, 80.0),
    n_grid: int = 80,
    refine_iters: int = 60,
) -> float:
    """
    Estimate per-user sigma (sigma_hat) from ONE experiment's observed daily trajectory x_t.

    Model:
      x_t = a + b exp(-(t-1)/lambda) + eps_t,
      eps_t ~ N(0, sigma^2 / N_eff)  with constant N_eff across t.

    Steps:
      1) Fit (a,b,lambda) by your existing fit_one_experiment_mle -> RSS over days.
      2) Residual variance estimate for eps_t: s2_eps = RSS/(T-2).
      3) Convert to per-user sigma_hat: sigma_hat^2 ≈ s2_eps * N_eff.

    Returns
    -------
    sigma_hat (float) on the per-user scale.
    """
    x = np.asarray(x, float)
    T = x.size
    if T < 3:
        raise ValueError("Need T>=3 to estimate sigma from residuals with df=T-2.")
    if N_eff <= 0:
        raise ValueError("N_eff must be positive.")

    fit = fit_one_experiment_mle(
        x,
        lambda_bounds=lambda_bounds,
        n_grid=n_grid,
        refine_iters=refine_iters,
    )
    rss = float(fit["rss"])
    s2_eps = rss / (T - 2)              # Var(eps_t) estimate
    sigma_hat = math.sqrt(s2_eps * N_eff)
    return float(sigma_hat)


def last_day_method_lower_bound_scipy(
    x: np.ndarray,
    *,
    N_eff: float,
    sigma_hat: float,
    alpha: float,
    eta_by_k: Dict[int, float],
    ks: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Implements your heuristic (NO Bonferroni correction):

        L_k = mean(x_{T-k+1:T})
              - ( eta(T,k,alpha) + z_{1-alpha/2} * sigma_hat / sqrt(k * N_eff) )

        return max_k L_k.

    Conventions:
    - x is length-T array of observed daily estimates (e.g. daily diff-in-means d_t).
    - mean(x_{T-k+1:T}) means the average of the LAST k entries (includes day T).
    - Default ks is 1..T-1 (matching your "k in [T-1]").

    Returns
    -------
    dict with:
      k_star, lower_bound, L_by_k, xbar_by_k, se_by_k, z
    """
    x = np.asarray(x, float)
    T = x.size
    if T < 2:
        raise ValueError("x must have length >= 2.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if N_eff <= 0:
        raise ValueError("N_eff must be positive.")
    if sigma_hat <= 0:
        raise ValueError("sigma_hat must be positive.")

    if ks is None:
        ks = np.arange(1, T)  # 1..T-1 by default
    else:
        ks = np.asarray(ks, int)

    for k in ks:
        if k < 1 or k > T:
            raise ValueError(f"Each k must be in [1, T]; got k={k}, T={T}.")
        if int(k) not in eta_by_k:
            raise ValueError(f"eta_by_k missing k={k}.")

    z = float(norm.ppf(1.0 - alpha / 2.0))

    L_by_k, xbar_by_k, se_by_k = {}, {}, {}
    for k in ks:
        k = int(k)
        xbar = float(np.mean(x[-k:]))
        se = float(sigma_hat / math.sqrt(k * N_eff))
        eta = float(eta_by_k[k])
        L = xbar - (eta + z * se)

        L_by_k[k] = L
        xbar_by_k[k] = xbar
        se_by_k[k] = se

    k_star = max(L_by_k, key=L_by_k.get)
    return {
        "k_star": int(k_star),
        "lower_bound": float(L_by_k[k_star]),
        "L_by_k": L_by_k,
        "xbar_by_k": xbar_by_k,
        "se_by_k": se_by_k,
        "z": z,
    }

def last_day_method_lower_bound_confseq_backward(
    x: np.ndarray,
    *,
    N_eff: float,
    sigma_hat: float,
    alpha: float,
    eta_by_k: Dict[int, float],
    ks: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Confidence-sequence-style backward selection of k (no Bonferroni):

        L_k = mean(x_{T-k+1:T})
              - ( eta_k
                  + (1/k) * sqrt( ( (sigma_hat^2 / N_eff) * k + rho )
                                 * log( ((sigma_hat^2 / N_eff) * k + rho) / (alpha^2 * rho) )
                                )
                )

    where rho = sigma_hat^2 * t0 and we set t0 = T (experiment length).

    Same arguments + return keys as last_day_method_lower_bound_scipy for drop-in compatibility.
    Note: we return `se_by_k` as the CS penalty term (the whole (1/k)*sqrt(...) piece),
          and `z` as NaN (not used in this construction).
    """
    x = np.asarray(x, float)
    T = x.size
    if T < 2:
        raise ValueError("x must have length >= 2.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if N_eff <= 0:
        raise ValueError("N_eff must be positive.")
    if sigma_hat <= 0:
        raise ValueError("sigma_hat must be positive.")

    if ks is None:
        ks = np.arange(1, T)  # 1..T-1
    else:
        ks = np.asarray(ks, int)

    for k in ks:
        if k < 1 or k >= T:
            raise ValueError(f"Each k must be in [1, T-1]; got k={k}, T={T}.")
        if int(k) not in eta_by_k:
            raise ValueError(f"eta_by_k missing k={k}.")

    sigma2 = float(sigma_hat * sigma_hat)
    t0 = T
    rho = sigma2 * t0  # convention: rho = sigma_hat^2 * t0

    L_by_k, xbar_by_k, se_by_k = {}, {}, {}

    for k in ks:
        k = int(k)
        xbar = float(np.mean(x[-k:]))

        # CS penalty term
        var_term = (sigma2 / float(N_eff)) * k + rho
        log_arg = var_term / (alpha * alpha * rho)
        if log_arg <= 1.0:
            # This shouldn't happen for typical alpha<1, but keep it safe.
            cs_pen = 0.0
        else:
            cs_pen = (1.0 / k) * math.sqrt(var_term * math.log(log_arg))

        eta = float(eta_by_k[k])
        L = xbar - (eta + cs_pen)

        L_by_k[k] = float(L)
        xbar_by_k[k] = float(xbar)
        se_by_k[k] = float(cs_pen)

    k_star = max(L_by_k, key=L_by_k.get)
    return {
        "k_star": int(k_star),
        "lower_bound": float(L_by_k[k_star]),
        "L_by_k": L_by_k,
        "xbar_by_k": xbar_by_k,
        "se_by_k": se_by_k,
        "z": float("nan"),
    }




# ============================================================
# 4) Experiment runner: test multiple priors and save results
# ============================================================

def run_experiment_with_prior(cfg: SyntheticABConfig, experiment_name: str = "default") -> Dict[str, Any]:
    """
    Run a complete experiment with a given prior configuration and return all results.
    
    Returns dict with:
      - true_hyperparams: the true prior parameters from cfg
      - fitted_hyperparams: the EB-fitted hyperparameters
      - sim_data: the simulated A/B test data
      - fits: per-experiment MLE results
      - etas: the delta_k thresholds
      - last_day_results: results from the last-day method on a held-out experiment
      - experiment_name: identifier for this configuration
    """
    # Generate synthetic data
    sim = generate_synthetic_ab_data(cfg)
    
    # Fit all experiments via MLE
    fits = fit_all_experiments_mle(sim["d_obs"], sim["N_eff"], lambda_bounds=(0.5, 80.0))
    
    # Estimate EB hyperparameters
    hyp = empirical_bayes_hyperparams(
        fits["a_hat"], fits["b_hat"], fits["lambda_hat"],
        nu_fixed=cfg.nu,
        scale_bounds=(1e-4, 2.0)
    )
    
    # Compute delta_k thresholds
    etas = delta_k_thresholds(
        T=cfg.T, alpha=0.05,
        pi=hyp["pi"], nu=hyp["nu"],
        sigma_pos=hyp["sigma_pos"], sigma_neg=hyp["sigma_neg"],
        loglambda_mean=hyp["loglambda_mean"], loglambda_sd=hyp["loglambda_sd"],
        n_mc=200_000, seed=1
    )
    
    # Apply last-day method to a held-out experiment
    i_new = 100
    x_new = sim["d_obs"][i_new]
    N_eff_new = float(sim["N_eff"][i_new])
    sigma_hat_new = estimate_sigma_hat_from_residuals(
        x_new, N_eff=N_eff_new, lambda_bounds=(0.5, 80.0), n_grid=80, refine_iters=60
    )
    
    res_last_day = last_day_method_lower_bound_confseq_backward(
        x_new,
        N_eff=N_eff_new,
        sigma_hat=sigma_hat_new,
        alpha=0.05,
        eta_by_k=etas,
    )
    
    return {
        "experiment_name": experiment_name,
        "true_hyperparams": {
            "a_mean": cfg.a_mean, "a_sd": cfg.a_sd,
            "pi": cfg.pi, "nu": cfg.nu,
            "sigma_pos": cfg.sigma_pos, "sigma_neg": cfg.sigma_neg,
            "lambda_low": cfg.lambda_low, "lambda_high": cfg.lambda_high,
            "K": cfg.K, "T": cfg.T,
            "N_T_range": (cfg.N_T_low, cfg.N_T_high),
            "N_C_range": (cfg.N_C_low, cfg.N_C_high),
        },
        "fitted_hyperparams": hyp,
        "sim": sim,
        "fits": fits,
        "etas": etas,
        "last_day_results": res_last_day,
        "held_out_experiment": {
            "i_new": i_new,
            "x_new": x_new,
            "N_eff_new": N_eff_new,
            "sigma_hat_new": sigma_hat_new,
            "true_last_day": float(sim["d_true"][i_new, -1]),
        }
    }


def save_experiment_results(results: Dict[str, Any], output_dir: str = "./experiment_results") -> None:
    """
    Save experiment results to a directory structure for later analysis and plotting.
    
    Creates:
      - {output_dir}/{experiment_name}/config.txt (hyperparameter summary)
      - {output_dir}/{experiment_name}/results.npy (pickled dict with all results)
      - {output_dir}/{experiment_name}/plots/ (placeholder for plots)
    """
    import os
    import pickle
    
    exp_name = results["experiment_name"]
    exp_dir = os.path.join(output_dir, exp_name)
    plots_dir = os.path.join(exp_dir, "plots")
    
    # Create directories
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save hyperparameter summary
    config_file = os.path.join(exp_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TRUE HYPERPARAMETERS (Prior):\n")
        for k, v in results["true_hyperparams"].items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\nFITTED HYPERPARAMETERS (EB Estimates):\n")
        for k, v in results["fitted_hyperparams"].items():
            if k != "counts":
                f.write(f"  {k}: {v}\n")
        f.write(f"  counts: {results['fitted_hyperparams']['counts']}\n")
        
        f.write("\nLAST-DAY METHOD RESULTS:\n")
        f.write(f"  Held-out experiment index: {results['held_out_experiment']['i_new']}\n")
        f.write(f"  N_eff: {results['held_out_experiment']['N_eff_new']:.2f}\n")
        f.write(f"  Estimated sigma: {results['held_out_experiment']['sigma_hat_new']:.4f}\n")
        f.write(f"  True last-day value: {results['held_out_experiment']['true_last_day']:.4f}\n")
        f.write(f"  k_star: {results['last_day_results']['k_star']}\n")
        f.write(f"  Lower bound: {results['last_day_results']['lower_bound']:.4f}\n")
    
    # Save all results as pickle for later reload
    results_file = os.path.join(exp_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Saved experiment results to {exp_dir}/")
    print(f"  - Config: {config_file}")
    print(f"  - Results: {results_file}")


# ============================================================
# 3) Example: run end-to-end
# ============================================================

if __name__ == "__main__":

    # Define multiple hand-picked priors to test
    priors_to_test = [
        {
            "name": "baseline_high_snr",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=3.0,
                pi=0.7, sigma_pos=0.10, sigma_neg=0.06,
                N_T_low=2000, N_T_high=20000,
                N_C_low=2000, N_C_high=20000,
            )
        },
        {
            "name": "low_effect_size",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=3.0,
                pi=0.7, sigma_pos=0.05, sigma_neg=0.03,  # Smaller effects
                N_T_low=2000, N_T_high=20000,
                N_C_low=2000, N_C_high=20000,
            )
        },
        {
            "name": "high_effect_size",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=3.0,
                pi=0.7, sigma_pos=0.20, sigma_neg=0.12,  # Larger effects
                N_T_low=2000, N_T_high=20000,
                N_C_low=2000, N_C_high=20000,
            )
        },
        {
            "name": "low_snr_small_samples",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=3.0,
                pi=0.7, sigma_pos=0.10, sigma_neg=0.06,
                N_T_low=500, N_T_high=1000,    # Much smaller samples
                N_C_low=500, N_C_high=1000,
            )
        },
        {
            "name": "more_negative_effects",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=3.0,
                pi=0.5, sigma_pos=0.10, sigma_neg=0.10,  # 50-50 split, equal scales
                N_T_low=2000, N_T_high=20000,
                N_C_low=2000, N_C_high=20000,
            )
        },
        {
            "name": "heavy_tailed_effects",
            "config": SyntheticABConfig(
                K=300, T=14, seed=0, nu=1.5,  # Heavier tails (lower nu)
                pi=0.7, sigma_pos=0.10, sigma_neg=0.06,
                N_T_low=2000, N_T_high=20000,
                N_C_low=2000, N_C_high=20000,
            )
        },
    ]
    
    # Run all experiments
    all_results = []
    for prior_spec in priors_to_test:
        print(f"\n{'='*60}")
        print(f"Running experiment: {prior_spec['name']}")
        print(f"{'='*60}")
        
        results = run_experiment_with_prior(prior_spec["config"], prior_spec["name"])
        all_results.append(results)
        
        # Print summary
        print(f"\nTrue hyperparams: {results['true_hyperparams']}")
        print(f"\nFitted hyperparams: {results['fitted_hyperparams']}")
        print(f"\nLast-day method (i_new={results['held_out_experiment']['i_new']}):")
        print(f"  k_star: {results['last_day_results']['k_star']}")
        print(f"  lower_bound: {results['last_day_results']['lower_bound']:.4f}")
        print(f"  true_last_day: {results['held_out_experiment']['true_last_day']:.4f}")
        
        # Save results
        save_experiment_results(results, output_dir="./experiment_results")
        
        # Generate and save plots
        print(f"\nGenerating plots for {prior_spec['name']}...")
        
        # Plot 1: etas thresholds
        plt.figure(figsize=(10, 6))
        ks = np.array(sorted(results['etas'].keys()))
        ys = np.array([results['etas'][k] for k in ks])
        plt.plot(ks, ys, marker="o", linewidth=2, markersize=8)
        plt.xlabel("k (window size)", fontsize=12)
        plt.ylabel(r"$\eta_k$ (bias correction threshold)", fontsize=12)
        plt.title(f"Delta_k thresholds (95% quantile) - {prior_spec['name']}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./experiment_results/{prior_spec['name']}/plots/etas_thresholds.png", dpi=150)
        plt.close()
        
        # Plot 2: Lower bounds by window size
        plt.figure(figsize=(10, 6))
        ks_plot = np.array(sorted(results['last_day_results']['L_by_k'].keys()))
        L_vals = np.array([results['last_day_results']['L_by_k'][k] for k in ks_plot])
        plt.plot(ks_plot, L_vals, marker="o", linewidth=2, markersize=8, label="L_k (lower bound)")
        plt.axhline(results['held_out_experiment']['true_last_day'], color='r', linestyle='--', linewidth=2, label=f"True value: {results['held_out_experiment']['true_last_day']:.4f}")
        plt.axvline(results['last_day_results']['k_star'], color='g', linestyle='--', linewidth=2, label=f"k_star: {results['last_day_results']['k_star']}")
        plt.xlabel("k (window size)", fontsize=12)
        plt.ylabel("Lower bound value", fontsize=12)
        plt.title(f"Last-day method lower bounds - {prior_spec['name']}", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./experiment_results/{prior_spec['name']}/plots/lower_bounds.png", dpi=150)
        plt.close()
        
        # Plot 3: Sample trajectories
        plot_sample_trajectories_aligned_start(results['sim'], n_plot=12, which="d_obs", seed=0)
        plt.savefig(f"./experiment_results/{prior_spec['name']}/plots/sample_trajectories_observed.png", dpi=150)
        plt.close()
        
        plot_sample_trajectories_aligned_start(results['sim'], n_plot=12, which="d_true", seed=0)
        plt.savefig(f"./experiment_results/{prior_spec['name']}/plots/sample_trajectories_true.png", dpi=150)
        plt.close()
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to ./experiment_results/")
    print(f"{'='*60}")
