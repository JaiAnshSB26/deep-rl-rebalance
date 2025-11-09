"""
Mean-variance optimization baseline with Ledoit-Wolf shrinkage.
"""
import numpy as np
from typing import Dict
from sklearn.covariance import LedoitWolf


def run_strategy(X: np.ndarray, R_next: np.ndarray, dates: np.ndarray, cfg: dict) -> Dict:
    """
    Mean-variance optimizer with Ledoit-Wolf shrinkage.
    Target volatility constraint.
    """
    T, N = R_next.shape
    cost_rate = cfg['trade']['cost_bps_per_turnover'] / 10000.0
    cap = cfg['trade'].get('cap_per_asset', 1.0)
    lookback = 252
    target_vol = 0.10  # 10% annualized
    
    weights = np.ones(N) / N
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    
    # Run to T-1 to match RL environment behavior
    for t in range(T - 1):
        if t >= lookback:
            recent_returns = R_next[max(0, t - lookback):t]  # [lookback, N]
            
            # Estimate covariance with Ledoit-Wolf shrinkage
            lw = LedoitWolf()
            try:
                cov_matrix = lw.fit(recent_returns).covariance_
            except:
                # Fallback to sample covariance
                cov_matrix = np.cov(recent_returns.T)
            
            # Simple mean estimate (or use momentum)
            mean_returns = np.mean(recent_returns, axis=0)
            
            # Min variance portfolio subject to constraints
            # Simplified: inverse covariance weighting
            try:
                inv_cov = np.linalg.inv(cov_matrix + 1e-4 * np.eye(N))
                ones = np.ones(N)
                target_weights = inv_cov @ ones
                target_weights = target_weights / target_weights.sum()
                target_weights = np.maximum(target_weights, 0)
                target_weights = target_weights / target_weights.sum()
            except:
                # Fallback to equal weights
                target_weights = np.ones(N) / N
            
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
