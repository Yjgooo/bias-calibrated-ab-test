
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

def simulate_treatment_gp(
    n_days=14,
    mu0=0.0,
    sigma_f=0.4,
    ell=4.0,
    n_per_day=3000,
    sigma_y=1.2,
    seed=0,
    alpha=1e-8,
):
    """
    Simulate treatment group latent true mean mu_B(t) as a GP with exponential kernel,
    and observed daily averages ybar_B with sampling noise.

    Returns
    -------
    t : (n_days,) int array of day indices (1..n_days)
    mu_B : (n_days,) float array, latent true mean
    ybar_B : (n_days,) float array, observed daily averages
    """
    rs = np.random.RandomState(seed)

    # time grid for sklearn (n_samples, n_features)
    t = np.arange(1, n_days + 1).reshape(-1, 1)

    # exponential kernel == Matern(nu=0.5)
    kernel = ConstantKernel(sigma_f**2) * Matern(length_scale=ell, nu=0.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=False)

    # sample latent true mean
    mu_B = mu0 + gp.sample_y(t, n_samples=1, random_state=rs).ravel()

    # sample observed daily averages (mean of n_per_day iid outcomes)
    ybar_B = mu_B + rs.normal(0.0, sigma_y / np.sqrt(n_per_day), size=n_days)

    return t.ravel(), mu_B, ybar_B


if __name__ == "__main__":
    t, mu_B, ybar_B = simulate_treatment_gp(seed=0) #t: date; mu_B: true mean; ybar_B: observed daily avg

    plt.figure()
    plt.plot(t, mu_B, marker="o", label="mu_B (true mean)")
    plt.plot(t, ybar_B, marker="x", linestyle="--", label="ybar_B (observed)")
    plt.xlabel("Day"); plt.ylabel("Value"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

    