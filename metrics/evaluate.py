"""
Performance evaluation metrics for portfolio strategies.
"""
import numpy as np
from typing import Dict


def performance_summary(daily_returns: np.ndarray, freq: int = 252) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Args:
        daily_returns: Array of daily returns
        freq: Trading days per year (default 252)
        
    Returns:
        Dictionary with CAGR, Vol, Sharpe, Sortino, Calmar, MaxDD, Tail ratio
    """
    if len(daily_returns) == 0:
        return {
            'CAGR': 0.0,
            'Volatility': 0.0,
            'Sharpe': 0.0,
            'Sortino': 0.0,
            'Calmar': 0.0,
            'Max_Drawdown': 0.0,
            'Tail_Ratio': 0.0
        }
    
    # CAGR
    cum_return = np.prod(1 + daily_returns)
    years = len(daily_returns) / freq
    cagr = (cum_return ** (1 / years)) - 1 if years > 0 else 0
    
    # Volatility (annualized)
    volatility = np.std(daily_returns, ddof=1) * np.sqrt(freq)
    
    # Sharpe ratio
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns, ddof=1) + 1e-10)) * np.sqrt(freq)
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(freq)
        sortino = (np.mean(daily_returns) * freq) / (downside_std + 1e-10)
    else:
        sortino = sharpe
    
    # Max Drawdown
    cum_returns = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns) / running_max
    max_drawdown = np.max(drawdown)
    
    # Calmar ratio
    calmar = cagr / (max_drawdown + 1e-10)
    
    # Tail ratio (95th percentile / 5th percentile)
    percentile_95 = np.percentile(daily_returns, 95)
    percentile_5 = np.percentile(daily_returns, 5)
    tail_ratio = percentile_95 / (abs(percentile_5) + 1e-10)
    
    return {
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max_Drawdown': max_drawdown,
        'Tail_Ratio': tail_ratio
    }


def turnover_and_cost(turnover: np.ndarray, costs: np.ndarray, freq: int = 252) -> Dict[str, float]:
    """
    Compute turnover and cost metrics.
    
    Args:
        turnover: Array of daily turnover values
        costs: Array of daily costs
        freq: Trading days per year
        
    Returns:
        Dictionary with annualized turnover and cost drag in bps
    """
    # Annualized turnover (average daily turnover * freq)
    avg_daily_turnover = np.mean(turnover)
    annualized_turnover = avg_daily_turnover * freq
    
    # Cost drag in bps per year
    avg_daily_cost = np.mean(costs)
    cost_drag_bps = avg_daily_cost * freq * 10000
    
    return {
        'Annualized_Turnover': annualized_turnover,
        'Cost_Drag_bps': cost_drag_bps
    }


def herfindahl_index(weights: np.ndarray) -> float:
    """
    Compute Herfindahl-Hirschman Index (concentration measure).
    
    Args:
        weights: [T, N] array of portfolio weights
        
    Returns:
        Average HHI across time
    """
    hhi = np.sum(weights ** 2, axis=1)
    return np.mean(hhi)


def full_evaluation(results: Dict, freq: int = 252) -> Dict[str, float]:
    """
    Compute all evaluation metrics from strategy results.
    
    Args:
        results: Dictionary from baseline/RL strategy with keys:
                 'daily_returns', 'weights', 'turnover', 'costs'
        freq: Trading days per year
        
    Returns:
        Dictionary with all metrics
    """
    perf = performance_summary(results['daily_returns'], freq)
    turn_cost = turnover_and_cost(results['turnover'], results['costs'], freq)
    hhi = herfindahl_index(results['weights'])
    
    return {
        **perf,
        **turn_cost,
        'HHI': hhi
    }


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate daily returns
    daily_returns = np.random.normal(0.0005, 0.01, 252)
    turnover = np.random.uniform(0.01, 0.05, 252)
    costs = turnover * 0.0025
    weights = np.random.dirichlet(np.ones(5), 252)
    
    results = {
        'daily_returns': daily_returns,
        'weights': weights,
        'turnover': turnover,
        'costs': costs
    }
    
    metrics = full_evaluation(results)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
