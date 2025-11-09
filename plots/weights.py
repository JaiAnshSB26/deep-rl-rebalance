"""
Portfolio weights visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_weights_heatmap(weights: np.ndarray, assets: list, savepath: str = None, dates=None):
    """
    Plot heatmap of portfolio weights over time.
    
    Args:
        weights: [T, N] array of weights
        assets: List of asset names
        savepath: Path to save figure
        dates: Optional dates for x-axis
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Transpose for plotting (assets on y-axis, time on x-axis)
    im = ax.imshow(weights.T, aspect='auto', cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_yticks(np.arange(len(assets)))
    ax.set_yticklabels(assets)
    ax.set_ylabel('Assets', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=12)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved weights heatmap to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_weights_area(weights: np.ndarray, assets: list, savepath: str = None, dates=None):
    """Plot stacked area chart of portfolio weights."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = dates if dates is not None else np.arange(weights.shape[0])
    
    ax.stackplot(x, *[weights[:, i] for i in range(weights.shape[1])], labels=assets, alpha=0.8)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_xlabel('Date' if dates is not None else 'Days', fontsize=12)
    ax.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved weights area plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_weight_statistics(weights: np.ndarray, assets: list, savepath: str = None):
    """Plot statistics of weights (mean, std, min, max per asset)."""
    mean_weights = np.mean(weights, axis=0)
    std_weights = np.std(weights, axis=0)
    min_weights = np.min(weights, axis=0)
    max_weights = np.max(weights, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(assets))
    width = 0.6
    
    ax.bar(x, mean_weights, width, yerr=std_weights, capsize=5, alpha=0.7, label='Mean ± Std')
    ax.scatter(x, min_weights, color='red', marker='_', s=100, label='Min')
    ax.scatter(x, max_weights, color='green', marker='_', s=100, label='Max')
    
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_title('Weight Statistics by Asset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved weight statistics plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    T, N = 252, 5
    weights = np.random.dirichlet(np.ones(N), T)
    assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'DBC']
    
    plot_weights_heatmap(weights, assets, savepath="figures/test_weights_heatmap.png")
    plot_weights_area(weights, assets, savepath="figures/test_weights_area.png")
    plot_weight_statistics(weights, assets, savepath="figures/test_weight_stats.png")
