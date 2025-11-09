"""
Momentum tilt baseline: overweight positive momentum, underweight negative.
"""
import numpy as np
from typing import Dict


def run_strategy(X: np.ndarray, R_next: np.ndarray, dates: np.ndarray, cfg: dict) -> Dict:
    """
    Momentum tilt: base equal weight, tilt based on 63-day momentum.
    """
    T, N = R_next.shape
    cost_rate = cfg['trade']['cost_bps_per_turnover'] / 10000.0
    cap = cfg['trade'].get('cap_per_asset', 1.0)
    mom_window = 63
    
    weights = np.ones(N) / N
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    
    # Run to T-1 to match RL environment behavior
    for t in range(T - 1):
        # Compute momentum signals
        if t >= mom_window:
            recent_returns = R_next[max(0, t - mom_window):t]  # [window, N]
            # Cumulative return over window
            cum_returns = np.sum(recent_returns, axis=0)
            
            # Normalize to [-1, 1]
            mom_signals = cum_returns / (np.abs(cum_returns).max() + 1e-6)
            
            # Base equal weight + momentum tilt
            # Positive momentum -> overweight, negative -> underweight
            base = 1.0 / N
            tilt_strength = 0.5  # Max tilt factor
            target_weights = base * (1.0 + tilt_strength * mom_signals)
            
            # Ensure non-negative
            target_weights = np.maximum(target_weights, 0)
            
            # Normalize
            target_weights = target_weights / (target_weights.sum() + 1e-10)
            
            # Apply cap
            if cap < 1.0:
                for _ in range(10):
                    overflow = np.maximum(target_weights - cap, 0).sum()
                    target_weights = np.minimum(target_weights, cap)
                    if overflow < 1e-6:
                        break
                    uncapped_mask = target_weights < cap
                    if uncapped_mask.sum() > 0:
                        target_weights[uncapped_mask] += overflow / uncapped_mask.sum()
                target_weights = target_weights / target_weights.sum()
        else:
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
