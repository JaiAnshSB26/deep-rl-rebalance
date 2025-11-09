"""
Equal weight buy-and-hold baseline.
"""
import numpy as np
from typing import Dict


def run_strategy(X: np.ndarray, R_next: np.ndarray, dates: np.ndarray, cfg: dict) -> Dict:
    """
    Equal weight buy-and-hold: Initial equal weights, no rebalancing.
    
    Returns:
        dict with 'daily_returns', 'weights', 'turnover', 'costs'
    """
    T, N = R_next.shape
    cost_rate = cfg['trade']['cost_bps_per_turnover'] / 10000.0
    
    # Initial equal weights
    weights = np.ones(N) / N
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    
    # Run to T-1 to match RL environment behavior
    for t in range(T - 1):  #T before..
        # No rebalancing - target weights = current weights
        target_weights = weights.copy()
        
        # Turnover is zero (no trades)
        turnover = 0.0
        cost = 0.0
        
        # Portfolio return
        gross_return = np.dot(weights, R_next[t])
        net_return = gross_return - cost
        
        # Update weights due to market movement
        post_return_weights = weights * (1.0 + R_next[t])
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
