# Bayesian A/B Testing with Adaptive Window Selection

A Python framework for Bayesian A/B testing that includes:
- Synthetic data generation with exponential decay treatment effect models
- Per-experiment Maximum Likelihood Estimation (MLE) for treatment effect parameters
- Empirical Bayes (EB) hyperparameter learning from historical experiments
- Adaptive lower bound estimation via the "last-day method"
- Modular architecture with separate simulation, estimation, and visualization modules

## Project Structure

```
bayesian-ab/
├── adaptive_estimator.py        # Core estimation pipeline
├── visualization.py              # Plotting utilities
├── experiments/                  # One-time exploratory scripts
├── experiment_results/           # Results from prior comparisons
└── README.md
```

## Key Features

### 1. Simulator (`generate_synthetic_ab_data`)
- Generates K independent A/B test experiments
- Models treatment effect decay: $d_t = a_i + b_i \cdot e^{-(t-1)/\lambda_i} + \text{noise}$
- Supports configurable prior distributions via `SyntheticABConfig`

### 2. Per-Experiment MLE (`fit_one_experiment_mle`, `fit_all_experiments_mle`)
- Profile likelihood estimation of exponential decay parameters (a, b, λ)
- Uses golden-section search for robust optimization
- Handles varying sample sizes per day

### 3. Empirical Bayes (`empirical_bayes_hyperparams`)
- Two-stage plug-in EB procedure
- Learns population-level hyperparameters from K experiments
- Estimates both continuous (Normal) and mixture priors

### 4. Adaptive Lower Bounds (`last_day_method_lower_bound_confseq_backward`)
- Confidence-sequence-style lower bounds for treatment effect
- Automatically selects optimal window size k
- Uses EB-learned bias correction thresholds

### 5. Multi-Prior Testing (`run_experiment_with_prior`, `save_experiment_results`)
- Test multiple hand-picked prior configurations
- Automatically saves all results (hyperparameters, data, plots)
- Generates comparative visualizations

## Usage

```python
from adaptive_estimator import SyntheticABConfig, run_experiment_with_prior, save_experiment_results

# Define a prior configuration
cfg = SyntheticABConfig(
    K=300, T=14, seed=0, nu=3.0,
    pi=0.7, sigma_pos=0.10, sigma_neg=0.06,
    N_T_low=2000, N_T_high=20000,
    N_C_low=2000, N_C_high=20000,
)

# Run the experiment
results = run_experiment_with_prior(cfg, "my_experiment")

# Save results and plots
save_experiment_results(results, output_dir="./experiment_results")
```

## Installation

Requires: NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

## Mathematical Background

### Treatment Effect Model
$$d_t = a_i + b_i \cdot e^{-(t-1)/\lambda_i} + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2/N_{\text{eff}})$$

### Prior Distribution for Effect Size
$$b_i = s_i \cdot h_i, \quad P(s_i = +1) = \pi, \quad h_i | s_i \sim \text{Half-t}(\nu, \sigma_s)$$

### Cumulative Effect (Delta_k)
$$\Delta_k = b \cdot r^{T-k} \left[ r^{k-1} - \frac{1-r^k}{k(1-r)} \right], \quad r = e^{-1/\lambda}$$

### Adaptive Lower Bound
$$L_k = \bar{x}_{T-k+1:T} - \left( \eta_k + \text{penalty}(k) \right)$$

where $\eta_k$ is the $(1-\alpha)$-quantile of $\Delta_k$ under the EB prior.

## References

- Exponential decay model: Common in online experimentation
- Empirical Bayes: Efron & Morris (1973)
- Confidence sequences: Ramdas et al. (2020)
