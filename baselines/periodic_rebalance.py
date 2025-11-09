"""
Periodic rebalancing to equal weights (monthly or quarterly).
"""
import numpy as np
import pandas as pd
from typing import Dict


def run_strategy(X: np.ndarray, R_next: np.ndarray, dates: np.ndarray, cfg: dict) -> Dict:
    """
    Periodically rebalance to equal weights (monthly or quarterly).
    """
    T, N = R_next.shape
    cost_rate = cfg['trade']['cost_bps_per_turnover'] / 10000.0
    freq = cfg['baselines'].get('periodic_rebalance_freq', 'monthly')
    
    weights = np.ones(N) / N
    target_weights = np.ones(N) / N
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    
    # Convert dates to pandas for easier date manipulation
    dates_pd = pd.to_datetime(dates)
    
    # Run to T-1 to match RL environment behavior
    for t in range(T - 1):
        # Check if we should rebalance
        rebalance = False
        if t == 0:
            rebalance = True
        elif freq == 'monthly':
            # Rebalance on first day of each month
            if dates_pd[t].month != dates_pd[t-1].month:
                rebalance = True
        elif freq == 'quarterly':
            # Rebalance on first day of each quarter
            if dates_pd[t].quarter != dates_pd[t-1].quarter:
                rebalance = True
        
        if rebalance:
            target_weights = np.ones(N) / N
        else:
            target_weights = weights.copy()
        
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
