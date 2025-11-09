"""
Equity curve and drawdown plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_equity_curve(daily_returns: np.ndarray, title: str = "Equity Curve", savepath: str = None, dates=None):
    """Plot cumulative equity curve with drawdown subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Compute equity
    equity = np.cumprod(1 + daily_returns)
    
    # X-axis
    if dates is not None:
        # Handle length mismatch: slice dates to match returns length
        x = dates[:len(daily_returns)]
    else:
        x = np.arange(len(equity))
    
    # Equity curve
    ax1.plot(x, equity, linewidth=2, label='Equity')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    
    ax2.fill_between(x, 0, -drawdown * 100, color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date' if dates is not None else 'Days', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved equity plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_equity_curves(results_dict: dict, title: str = "Strategy Comparison", savepath: str = None):
    """Plot multiple equity curves on the same chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy_name, results in results_dict.items():
        daily_returns = results['daily_returns']
        equity = np.cumprod(1 + daily_returns)
        ax.plot(equity, linewidth=2, label=strategy_name, alpha=0.8)
    
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test equity plot
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.01, 252)
    
    plot_equity_curve(daily_returns, title="Test Strategy", savepath="figures/test_equity.png")
