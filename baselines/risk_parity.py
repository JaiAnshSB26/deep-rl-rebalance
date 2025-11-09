"""
Risk parity baseline: inverse volatility weights.
"""
import numpy as np
from typing import Dict


def run_strategy(X: np.ndarray, R_next: np.ndarray, dates: np.ndarray, cfg: dict) -> Dict:
    """
    Risk parity: weights inversely proportional to volatility.
    """
    T, N = R_next.shape
    cost_rate = cfg['trade']['cost_bps_per_turnover'] / 10000.0
    cap = cfg['trade'].get('cap_per_asset', 1.0)
    vol_window = 63
    
    weights = np.ones(N) / N
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    
    # Run to T-1 to match RL environment behavior
    for t in range(T - 1):
        # Estimate volatilities from trailing window
        if t >= vol_window:
            recent_returns = R_next[max(0, t - vol_window):t]  # [window, N]
            vols = np.std(recent_returns, axis=0, ddof=1)
            vols = np.maximum(vols, 1e-6)  # Avoid division by zero
            
            # Inverse vol weights
            inv_vols = 1.0 / vols
            target_weights = inv_vols / inv_vols.sum()
            
            # Apply cap
            if cap < 1.0:
                for _ in range(10):  # Iterative capping
                    overflow = np.maximum(target_weights - cap, 0).sum()
                    target_weights = np.minimum(target_weights, cap)
                    if overflow < 1e-6:
                        break
                    uncapped_mask = target_weights < cap
                    if uncapped_mask.sum() > 0:
                        target_weights[uncapped_mask] += overflow / uncapped_mask.sum()
                target_weights = target_weights / target_weights.sum()
        else:
            # Not enough history, use equal weights
            target_weights = np.ones(N) / N
        
        # Turnover
        turnover = 0.5 * np.sum(np.abs(target_weights - weights))
        cost = cost_rate * turnover
        
        # Portfolio return
        gross_return = np.dot(target_weights, R_next[t])
        net_return = gross_return - cost
        
        # Update weights
        post_return_weights = target_weights * (1.0 + R_next[t])
        weights = post_return_weights / (post_return_weights.sum() + 1e-10)
        
        daily_returns.append(net_return)
        weights_history.append(target_weights)
        turnover_history.append(turnover)
        costs_history.append(cost)
    
    return {
        'daily_returns': np.array(daily_returns),
        'weights': np.array(weights_history),
        'turnover': np.array(turnover_history),
        'costs': np.array(costs_history)
    }
