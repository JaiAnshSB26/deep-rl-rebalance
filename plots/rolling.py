"""
Rolling metrics plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_rolling_sharpe(daily_returns: np.ndarray, window: int = 63, savepath: str = None, dates=None):
    """Plot rolling Sharpe ratio."""
    returns_series = pd.Series(daily_returns)
    roll_mean = returns_series.rolling(window, min_periods=window // 2).mean()
    roll_std = returns_series.rolling(window, min_periods=window // 2).std(ddof=1)
    roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle length mismatch: slice dates to match returns length
    x = dates[:len(daily_returns)] if dates is not None else np.arange(len(roll_sharpe))
    ax.plot(x, roll_sharpe.values, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel(f'Rolling Sharpe ({window}d)', fontsize=12)
    ax.set_xlabel('Date' if dates is not None else 'Days', fontsize=12)
    ax.set_title(f'Rolling Sharpe Ratio ({window}-day window)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved rolling Sharpe plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_drawdown(daily_returns: np.ndarray, savepath: str = None, dates=None):
    """Plot drawdown over time."""
    equity = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle length mismatch: slice dates to match returns length
    x = dates[:len(daily_returns)] if dates is not None else np.arange(len(drawdown))
    ax.fill_between(x, 0, -drawdown * 100, color='red', alpha=0.4)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date' if dates is not None else 'Days', fontsize=12)
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved drawdown plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_rolling_volatility(daily_returns: np.ndarray, window: int = 21, savepath: str = None, dates=None):
    """Plot rolling volatility (annualized)."""
    returns_series = pd.Series(daily_returns)
    roll_vol = returns_series.rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(252)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle length mismatch: slice dates to match returns length
    x = dates[:len(daily_returns)] if dates is not None else np.arange(len(roll_vol))
    ax.plot(x, roll_vol.values * 100, linewidth=2, color='darkorange')
    ax.set_ylabel(f'Rolling Volatility (%, annualized)', fontsize=12)
    ax.set_xlabel('Date' if dates is not None else 'Days', fontsize=12)
    ax.set_title(f'Rolling Volatility ({window}-day window)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved rolling volatility plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.01, 252)
    
    plot_rolling_sharpe(daily_returns, savepath="figures/test_rolling_sharpe.png")
    plot_drawdown(daily_returns, savepath="figures/test_drawdown.png")
