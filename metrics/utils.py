"""
Utility functions for metrics calculations.
"""
import numpy as np
import pandas as pd


def compute_equity_curve(daily_returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """Compute cumulative equity curve from daily returns."""
    return initial_value * np.cumprod(1 + daily_returns)


def compute_drawdown_series(daily_returns: np.ndarray) -> np.ndarray:
    """Compute drawdown series from daily returns."""
    cum_returns = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns) / running_max
    return drawdown


def rolling_sharpe(daily_returns: np.ndarray, window: int = 63, freq: int = 252) -> np.ndarray:
    """Compute rolling Sharpe ratio."""
    returns_series = pd.Series(daily_returns)
    roll_mean = returns_series.rolling(window, min_periods=window // 2).mean()
    roll_std = returns_series.rolling(window, min_periods=window // 2).std(ddof=1)
    roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(freq)
    return roll_sharpe.values


def rolling_volatility(daily_returns: np.ndarray, window: int = 21, freq: int = 252) -> np.ndarray:
    """Compute rolling volatility (annualized)."""
    returns_series = pd.Series(daily_returns)
    roll_vol = returns_series.rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(freq)
    return roll_vol.values


def annualize_return(total_return: float, years: float) -> float:
    """Annualize a total return."""
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0


def annualize_volatility(daily_vol: float, freq: int = 252) -> float:
    """Annualize daily volatility."""
    return daily_vol * np.sqrt(freq)
