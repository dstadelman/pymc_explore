import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from multiprocessing import freeze_support
import os
from tqdm import tqdm
import pytensor

# Check PyTensor configuration
print(f"PyTensor config.cxx: {pytensor.config.cxx}")

# Create output directory
output_dir = 'pymc_exploration'
os.makedirs(output_dir, exist_ok=True)

# Function to compute RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to generate datasets
def generate_datasets(num_samples=4096):
    try:
        # Define the quadratic function
        def quadratic_function(x):
            return -(1/50) * x**2 + 2 * x + 50

        # Generate x values (uniformly sampled from 0 to 100)
        np.random.seed(42)
        x_values = np.random.uniform(0, 100, num_samples)

        # Sort x_values for consistent plotting
        x_values = np.sort(x_values)

        # 1. Exact Dataset
        y_exact = quadratic_function(x_values)
        exact_dataset = pd.DataFrame({'x': x_values, 'y': y_exact})
        exact_dataset.to_csv(os.path.join(output_dir, 'exact_dataset.csv'), index=False)

        # 2. Noisy Dataset (Normal noise with std dev 10)
        noise = np.random.normal(loc=0, scale=10, size=num_samples)
        y_noisy = quadratic_function(x_values) + noise
        noisy_dataset = pd.DataFrame({'x': x_values, 'y': y_noisy})
        noisy_dataset.to_csv(os.path.join(output_dir, 'noisy_dataset.csv'), index=False)

        # 3. Random Dataset (y is random between 0 and 200)
        y_random = np.random.uniform(low=0, high=200, size=num_samples)
        random_dataset = pd.DataFrame({'x': x_values, 'y': y_random})
        random_dataset.to_csv(os.path.join(output_dir, 'random_dataset.csv'), index=False)

        # 4. Half Random, Half Noisy Dataset
        y_half_half = np.zeros(num_samples)
        choices = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])  # 50% noisy, 50% random
        for i in range(num_samples):
            if choices[i] == 0:  # Use noisy value
                y_half_half[i] = y_noisy[i]
            else:  # Use random value
                y_half_half[i] = y_random[i]
        half_half_dataset = pd.DataFrame({'x': x_values, 'y': y_half_half})
        half_half_dataset.to_csv(os.path.join(output_dir, 'half_half_dataset.csv'), index=False)

        # Print head of each dataset
        datasets = {
            'Exact': exact_dataset,
            'Noisy': noisy_dataset,
            'Random': random_dataset,
            'Half_Half': half_half_dataset
        }
        with open(os.path.join(output_dir, 'summary.txt'), 'a') as f:
            print("\nDataset Heads:", file=f)
            for name, df in datasets.items():
                print(f"\n{name} Dataset Head (first 5 rows):")
                print(df.head().to_string())
                print(f"\n{name} Dataset Head (first 5 rows):", file=f)
                print(df.head().to_string(), file=f)

        return datasets
    except Exception as e:
        print(f"Error generating datasets: {e}")
        with open(os.path.join(output_dir, 'summary.txt'), 'a') as f:
            print(f"Error generating datasets: {e}", file=f)
        raise

# Function to analyze a dataset
def analyze_dataset(dataset_name, data, summary_file):
    print(f"\nAnalyzing {dataset_name}...")
    with open(summary_file, 'a') as f:
        print(f"\nAnalyzing {dataset_name}...", file=f)
    
    try:
        x = data['x'].values
        y = data['y'].values

        # Scale x-values for numerical stability
        x_scaled = x / 100

        # Define the true quadratic function for comparison (using original x)
        def true_quadratic(x):
            return -(1/50) * x**2 + 2 * x + 50

        # Bayesian model
        with pm.Model() as model:
            # Priors for scaled model (a = -200, b = 200, c = 50)
            a = pm.Normal('a', mu=-200, sigma=20)    # Quadratic coefficient
            b = pm.Normal('b', mu=200, sigma=20)     # Linear coefficient
            c = pm.Normal('c', mu=50, sigma=10)      # Intercept
            sigma = pm.HalfNormal('sigma', sigma=20) # Noise standard deviation

            # Expected value of y, using scaled x
            mu = a * x_scaled**2 + b * x_scaled + c

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Sample prior
            prior = pm.sample_prior_predictive(samples=1000, random_seed=42)

            # Sample posterior
            with tqdm(total=4000, desc=f"Sampling {dataset_name}") as pbar:
                trace = pm.sample(
                    draws=1000,
                    tune=1000,
                    chains=4,
                    return_inferencedata=True,
                    random_seed=42,
                    idata_kwargs={"log_likelihood": True},
                    max_treedepth=15,
                    target_accept=0.99,
                    init='adapt_diag',
                    progressbar=False  # Managed by tqdm
                )
                pbar.update(4000)

            # Sample posterior predictive
            ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=True)

        # Extract posterior means
        a_mean = trace.posterior['a'].mean().values
        b_mean = trace.posterior['b'].mean().values
        c_mean = trace.posterior['c'].mean().values
        y_pred = a_mean * x_scaled**2 + b_mean * x_scaled + c_mean

        # Compute metrics
        mae = mean_absolute_error(y, y_pred)
        rmse_val = rmse(y, y_pred)
        r2 = r2_score(y, y_pred)
        loo = az.loo(trace, pointwise=True)

        # Check convergence diagnostics
        rhat = az.rhat(trace)
        max_rhat = max(rhat[v].max().values for v in rhat.data_vars if v in ['a', 'b', 'c', 'sigma'])
        ess = az.ess(trace)
        min_ess = min(ess[v].min().values for v in ess.data_vars if v in ['a', 'b', 'c', 'sigma'])

        # Print and save metrics
        metrics_str = f"""Posterior Mean Parameters:
  a: {a_mean:.4f} (True: -200.0000)
  b: {b_mean:.4f} (True: 200.0000)
  c: {c_mean:.4f} (True: 50.0000)
  sigma: {trace.posterior['sigma'].mean().values:.4f}
Metrics:
  MAE: {mae:.4f}
  RMSE: {rmse_val:.4f}
  R-squared: {r2:.4f}
  LOO: {loo.elpd_loo:.4f}
Convergence Diagnostics:
  Max R-hat: {max_rhat:.4f} (should be < 1.01 for convergence)
  Min ESS: {min_ess:.0f} (should be > 100 for reliability)
"""
        print(metrics_str)
        with open(summary_file, 'a') as f:
            print(metrics_str, file=f)

        # Plot prior and posterior distributions separately
        var_names = ['a', 'b', 'c', 'sigma']
        # Prior plot
        plt.figure(figsize=(12, 8))
        for i, var in enumerate(var_names):
            plt.subplot(2, 2, i+1)
            az.plot_dist(prior.prior[var], label='Prior')
            plt.title(f"{var} Prior")
            plt.xlabel(var)
            plt.ylabel('Density')
        plt.suptitle(f"{dataset_name}: Prior Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_prior.png"))
        plt.close()

        # Posterior plot (2 by 2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, var in enumerate(var_names):
            ax = axes[i // 2, i % 2]
            az.plot_posterior(trace, var_names=[var], ax=ax)
            ax.set_title(f"{var} Posterior")
        plt.suptitle(f"{dataset_name}: Posterior Distributions")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_posterior.png"))
        plt.close()

        # Trace plot for diagnostics
        plt.figure(figsize=(12, 8))
        az.plot_trace(trace, var_names=var_names)
        plt.suptitle(f"{dataset_name}: Trace Plots")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_trace.png"))
        plt.close()

        # Plot posterior predictive check
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label="Observed", alpha=0.2, s=10)  # Use original x
        plt.plot(x, y_pred, color='red', label="Predicted Mean", linewidth=2)
        y_ppc = ppc.posterior_predictive['y_obs'].values
        y_ppc_hdi = az.hdi(y_ppc, hdi_prob=0.95)
        plt.fill_between(x, y_ppc_hdi[:, 0], y_ppc_hdi[:, 1], color='red', alpha=0.1, label="95% Credible Interval")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{dataset_name}: Posterior Predictive Check")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_ppc.png"))
        plt.close()

        # Plot true vs predicted quadratic (using original x)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label="Observed", alpha=0.2, s=10)
        plt.plot(x, y_pred, color='red', label="Predicted Mean", linewidth=2)
        if dataset_name in ['Exact', 'Noisy']:  # Only plot true function for Exact and Noisy
            y_true = true_quadratic(x)
            plt.plot(x, y_true, color='green', label="True Quadratic", linewidth=2, linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{dataset_name}: True vs Predicted Quadratic")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_true_vs_pred.png"))
        plt.close()

        return trace, prior, ppc, {'mae': mae, 'rmse': rmse_val, 'r2': r2, 'loo': loo.elpd_loo}
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        with open(summary_file, 'a') as f:
            print(f"Error analyzing {dataset_name}: {e}", file=f)
        raise

# Main function
def main():
    try:
        # Generate datasets with 4096 samples
        datasets = generate_datasets(num_samples=4096)

        # Initialize summary file
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            print("Bayesian Inference Analysis Summary", file=f)
            print(f"Dataset Size: 4096 samples", file=f)
            print(f"Sampling: 4 chains, 1000 tune, 1000 draws per chain (4000 draws total)", file=f)

        # Analyze each dataset
        results = {}
        for name, data in datasets.items():
            trace, prior, ppc, metrics = analyze_dataset(name, data, summary_file)
            results[name] = {'trace': trace, 'prior': prior, 'ppc': ppc, 'metrics': metrics}

        # Summary comparison
        summary_str = "\nSummary of Metrics Across Datasets:\n"
        metrics_df = pd.DataFrame({
            name: result['metrics'] for name, result in results.items()
        }).T
        summary_str += metrics_df[['mae', 'rmse', 'r2', 'loo']].to_string()
        print(summary_str)
        with open(summary_file, 'a') as f:
            print(summary_str, file=f)
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    except Exception as e:
        print(f"Error in main: {e}")
        with open(os.path.join(output_dir, 'summary.txt'), 'a') as f:
            print(f"Error in main: {e}", file=f)
        raise

if __name__ == '__main__':
    if os.name == 'nt':  # Check if the OS is Windows
        freeze_support()  # Helps with multiprocessing on Windows
    main()