"""
Sensitivity analysis plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_cost_sweep(df_results: pd.DataFrame, savepath: str = None):
    """
    Plot Sharpe ratio vs transaction cost.
    
    Args:
        df_results: DataFrame with columns ['cost_bps', 'strategy', 'sharpe']
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_results['strategy'].unique()
    
    for strategy in strategies:
        data = df_results[df_results['strategy'] == strategy]
        ax.plot(data['cost_bps'], data['sharpe'], marker='o', linewidth=2, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Transaction Cost (bps per turnover)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Sharpe Ratio Sensitivity to Transaction Costs', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved cost sweep plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_risk_penalty_sweep(df_results: pd.DataFrame, savepath: str = None):
    """
    Plot Sharpe vs Max Drawdown for different risk penalties.
    
    Args:
        df_results: DataFrame with columns ['lambda_risk', 'sharpe', 'max_dd']
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot efficient frontier
    ax.scatter(df_results['max_dd'] * 100, df_results['sharpe'], 
               c=df_results['lambda_risk'], cmap='viridis', s=100, alpha=0.7)
    
    # Add annotations for lambda values
    for idx, row in df_results.iterrows():
        ax.annotate(f"λ={row['lambda_risk']:.1f}", 
                   (row['max_dd'] * 100, row['sharpe']),
                   textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Max Drawdown (%)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Risk-Return Tradeoff (Risk Penalty Sensitivity)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Risk Penalty (λ)', fontsize=12)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved risk penalty sweep plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_turnover_vs_sharpe(results_dict: dict, savepath: str = None):
    """
    Scatter plot of turnover vs Sharpe for different strategies.
    
    Args:
        results_dict: Dict mapping strategy name to results dict
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy_name, results in results_dict.items():
        # Compute metrics
        daily_returns = results['daily_returns']
        turnover = results['turnover']
        
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns, ddof=1) + 1e-10)) * np.sqrt(252)
        avg_turnover = np.mean(turnover) * 252  # Annualized
        
        ax.scatter(avg_turnover, sharpe, s=150, alpha=0.7, label=strategy_name)
        ax.annotate(strategy_name, (avg_turnover, sharpe), 
                   textcoords="offset points", xytext=(5,5), fontsize=9)
    
    ax.set_xlabel('Annualized Turnover', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Turnover vs Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved turnover vs sharpe plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


def plot_pareto_frontier(df_results: pd.DataFrame, x_metric: str = 'volatility', y_metric: str = 'cagr', savepath: str = None):
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        df_results: DataFrame with strategy results
        x_metric: X-axis metric (e.g., 'volatility', 'max_dd')
        y_metric: Y-axis metric (e.g., 'cagr', 'sharpe')
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'strategy' in df_results.columns:
        for strategy in df_results['strategy'].unique():
            data = df_results[df_results['strategy'] == strategy]
            ax.scatter(data[x_metric], data[y_metric], s=100, alpha=0.7, label=strategy)
    else:
        ax.scatter(df_results[x_metric], df_results[y_metric], s=100, alpha=0.7)
    
    ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{y_metric.upper()} vs {x_metric.upper()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved Pareto frontier plot to {savepath}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test cost sweep plot
    df_cost = pd.DataFrame({
        'cost_bps': [0, 5, 10, 20, 30, 50] * 2,
        'strategy': ['RL'] * 6 + ['Equal Weight'] * 6,
        'sharpe': [1.8, 1.7, 1.6, 1.4, 1.2, 0.9, 1.2, 1.15, 1.1, 1.0, 0.9, 0.7]
    })
    
    plot_cost_sweep(df_cost, savepath="figures/test_cost_sweep.png")
