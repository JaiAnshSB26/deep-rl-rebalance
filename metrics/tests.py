"""
Statistical tests for comparing strategies.
"""
import numpy as np
from scipy import stats
from typing import Dict


def diebold_mariano(x: np.ndarray, y: np.ndarray, loss: str = "neg_return", h: int = 1) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Args:
        x: Returns from strategy X
        y: Returns from strategy Y
        loss: Loss function ('neg_return' or 'squared')
        h: Forecast horizon (for computing Newey-West robust std)
        
    Returns:
        Dictionary with DM statistic and p-value
    """
    n = len(x)
    
    # Compute loss differential
    if loss == "neg_return":
        # Higher return is better, so loss = -return
        d = -x + y  # Positive d means x is better
    elif loss == "squared":
        d = x**2 - y**2
    else:
        raise ValueError(f"Unknown loss function: {loss}")
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Newey-West variance estimator
    def newey_west_var(d, h):
        n = len(d)
        gamma_0 = np.var(d, ddof=1)
        
        var = gamma_0
        for lag in range(1, h):
            gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
            weight = 1 - lag / (h + 1)
            var += 2 * weight * gamma_lag
        
        return var / n
    
    var_d = newey_west_var(d, h)
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d + 1e-10)
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        'DM_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'interpretation': 'X better than Y' if dm_stat > 0 else 'Y better than X'
    }


def sharpe_block_bootstrap(returns: np.ndarray, block: int = 20, reps: int = 5000, freq: int = 252) -> Dict:
    """
    Block bootstrap for Sharpe ratio confidence interval.
    
    Args:
        returns: Daily returns
        block: Block size (approx. 20 trading days)
        reps: Number of bootstrap replications
        freq: Trading days per year
        
    Returns:
        Dictionary with confidence intervals
    """
    n = len(returns)
    sharpe_boots = []
    
    for _ in range(reps):
        # Moving block bootstrap
        boot_returns = []
        while len(boot_returns) < n:
            start_idx = np.random.randint(0, max(1, n - block))
            boot_returns.extend(returns[start_idx:start_idx + block])
        
        boot_returns = np.array(boot_returns[:n])
        
        # Compute Sharpe
        if len(boot_returns) > 0:
            sharpe = (np.mean(boot_returns) / (np.std(boot_returns, ddof=1) + 1e-10)) * np.sqrt(freq)
            sharpe_boots.append(sharpe)
    
    sharpe_boots = np.array(sharpe_boots)
    
    # Confidence intervals
    ci_lower = np.percentile(sharpe_boots, 2.5)
    ci_upper = np.percentile(sharpe_boots, 97.5)
    
    # Observed Sharpe
    observed_sharpe = (np.mean(returns) / (np.std(returns, ddof=1) + 1e-10)) * np.sqrt(freq)
    
    return {
        'observed_sharpe': observed_sharpe,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'bootstrap_mean': np.mean(sharpe_boots),
        'bootstrap_std': np.std(sharpe_boots, ddof=1)
    }


if __name__ == "__main__":
    # Test statistical tests
    np.random.seed(42)
    
    # Simulate two strategies
    returns_x = np.random.normal(0.0006, 0.01, 252)
    returns_y = np.random.normal(0.0004, 0.012, 252)
    
    # DM test
    dm_result = diebold_mariano(returns_x, returns_y)
    print("Diebold-Mariano Test:")
    for key, value in dm_result.items():
        print(f"  {key}: {value}")
    
    # Block bootstrap
    boot_result = sharpe_block_bootstrap(returns_x)
    print("\nBlock Bootstrap (Strategy X):")
    for key, value in boot_result.items():
        print(f"  {key}: {value:.4f}")
