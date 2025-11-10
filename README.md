# Deep Reinforcement Learning for Multi-Asset Portfolio Rebalancing

[![Project PDF Report](https://img.shields.io/badge/Research-PDF-red)](RESEARCH_REPORT.pdf)


[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade deep reinforcement learning project/agent/model for learning cost-aware portfolio rebalancing policies using Proximal Policy Optimization (PPO). This system explicitly models transaction costs and compares against strong quantitative baselines including mean-variance optimization, risk parity, and momentum strategies.

---

## Overview

Traditional portfolio optimization methods suffer from parameter estimation error and fail to adapt to changing market conditions. This project explores whether deep reinforcement learning can learn dynamic, cost-aware rebalancing strategies that account for transaction frictions.

**Key Features:**
- ✅ **Explicit Transaction Cost Modeling**: 25 bps per turnover in reward function
- ✅ **Causal Feature Engineering**: Strict no-lookahead guarantee (18 features per asset)
- ✅ **Multiple RL Algorithms**: PPO (primary), SAC (secondary)
- ✅ **Strong Baselines**: Equal-weight, periodic rebalance, risk parity, momentum, mean-variance
- ✅ **Statistical Rigor**: Diebold-Mariano tests, block bootstrap confidence intervals
- ✅ **Reproducible**: Seeded RNG, configuration management, versioned dependencies

---

## Results Summary

**Test Period**: 2022-2025 (4 years out-of-sample, includes 2022 bear market)

| Strategy | Sharpe | CAGR | MaxDD | Turnover | Cost Drag |
|----------|--------|------|-------|----------|-----------|
| **PPO (RL)** | **0.33** | **3.5%** | **29%** | **10.6%** | **265 bps** |
| Equal Weight | **0.55** | 6.2% | 21% | 0.0% | 0 bps |
| Periodic Rebal | 0.51 | 5.7% | 22% | 0.15% | 3.7 bps |
| Risk Parity | 0.49 | 5.0% | 22% | 1.4% | 34 bps |
| Momentum | 0.40 | 4.3% | 23% | 3.6% | 90 bps |
| Mean-Variance | 0.29 | 2.6% | 22% | 1.2% | 29 bps |

**Key Findings:**
- RL agent learned **cost-conscious behavior** (10.6% turnover vs 100%+ typical for active strategies)
- Agent exhibited **defensive tilts** (overweight bonds during 2022 crash, reduced tech exposure during volatility)
- At zero transaction costs, RL Sharpe = **0.53** (matching best baseline)
- Statistical tests show performance is **comparable** to best baseline (DM test p-value = 0.237)

The underperformance during 2022-2025 reflects that static diversification worked well in this specific inflation/rate-hike regime, demonstrating the challenge of distribution shift in financial ML.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/JaiAnshSB26/deep-rl-rebalance.git
cd deep-rl-rebalance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

The easiest way to reproduce results is via the Jupyter notebook:

```bash
jupyter notebook notebooks/01_train_evaluate.ipynb
```

Run all cells to:
1. Download market data (yfinance)
2. Engineer causal features
3. Train PPO agent (~30-60 min on CPU)
4. Evaluate on test set
5. Generate visualizations and statistical tests

---

## 📁 Project Structure

```
deep-rl-rebalance/
├── config.yaml               # All hyperparameters and settings
├── requirements.txt          # Python dependencies
├── RESEARCH_REPORT.md        # Full technical report
│
├── data/
│   ├── download.py           # Fetch OHLCV from yfinance
│   ├── features.py           # Causal feature engineering
│   └── splits.py             # Train/valid/test temporal splits
│
├── envs/
│   └── portfolio_env.py      # Gymnasium-compatible MDP
│
├── agents/
│   ├── ppo_trainer.py        # PPO with validation checkpointing
│   └── sac_trainer.py        # Soft Actor-Critic (alternative)
│
├── baselines/
│   ├── equal_weights.py      # Buy-and-hold equal weight
│   ├── periodic_rebalance.py # Monthly/quarterly rebalancing
│   ├── risk_parity.py        # Inverse volatility weighting
│   ├── momentum_tilt.py      # Trend-following tilts
│   └── mean_variance.py      # Markowitz with Ledoit-Wolf
│
├── metrics/
│   ├── evaluate.py           # Performance metrics (Sharpe, Sortino, etc.)
│   ├── tests.py              # Statistical tests (DM, bootstrap)
│   └── utils.py              # Utility functions
│
├── plots/
│   ├── equity.py             # Equity curves and drawdowns
│   ├── rolling.py            # Rolling performance metrics
│   ├── weights.py            # Portfolio weight analysis
│   └── sensitivity.py        # Parameter sensitivity sweeps
│
├── notebooks/
│   └── 01_train_evaluate.ipynb  # End-to-end pipeline
│
└── results/                  # Generated outputs (gitignored)
    ├── *.csv                 # Performance data
    ├── artifacts.json        # Experiment metadata
    └── figures/              # Visualizations (9 plots)
```

---

## Methodology

### Environment Design

**State Space**:
- Feature matrix: 9 assets × 18 features (momentum, volatility, technical indicators)
- Current portfolio weights
- Rolling portfolio volatility
- PCA factors of covariance matrix
- Market context (VIX level/change)

**Action Space**: 
- Raw logits → softmax → long-only weights
- Constraints: sum = 1, max 30% per asset

**Reward Function**:
```
R_t = net_return_t - λ * volatility_t - α * drawdown_increment_t
```
Where net return accounts for transaction costs (25 bps per unit turnover).

**Transition Dynamics**:
- Turnover = 0.5 × Σ|w_target - w_current|
- Cost = cost_rate × turnover
- Weights drift with market returns, then renormalize

### Features (18 per asset)

All features use only past data to ensure causality:

| Category | Features | Lookback |
|----------|----------|----------|
| **Momentum** | lag_1d, lag_2d, lag_5d | 1-5 days |
| **Trend** | mean_5d, mean_21d, mean_63d | 5-63 days |
| **Volatility** | std_5d, std_21d, std_63d | 5-63 days |
| **Technical** | RSI-14, MACD, Bollinger Bands, ATR-14 | 14-26 days |
| **Market** | VIX level, VIX 5d change, SPY return | Current |

**Normalization**: Cross-sectional winsorization (5%, 95%) + z-scoring within each date to prevent lookahead bias.

### Training Details

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Network**: MLP [256, 256] with Tanh activation
- **Training Steps**: 800,000 (~30-60 min on CPU)
- **Validation**: Every 100 updates, save best Sharpe model
- **Observation Normalization**: Running mean/std with ±10σ clipping

**Hyperparameters**:
```yaml
n_steps: 512
batch_size: 256
gamma: 0.99
learning_rate: 3e-4
ent_coef: 0.005
clip_range: 0.2
```

### Baselines

All baselines use identical transaction cost modeling (25 bps) for fair comparison:

1. **Equal Weight Buy & Hold**: 1/N, no rebalancing
2. **Periodic Rebalancing**: Monthly rebalance to 1/N
3. **Risk Parity**: Inverse volatility weighting (63-day lookback)
4. **Momentum Tilt**: Equal weight + tilt based on 63-day momentum
5. **Mean-Variance**: Markowitz optimization with Ledoit-Wolf covariance shrinkage

---

## Evaluation Metrics

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **CAGR**: Compound annual growth rate
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: CAGR / Max Drawdown
- **Max Drawdown**: Largest peak-to-trough decline
- **Tail Ratio**: 95th / |5th| percentile returns

### Trading Metrics
- **Annualized Turnover**: Average daily turnover × 252
- **Cost Drag**: Average daily cost × 252 × 10,000 bps

### Statistical Tests
- **Diebold-Mariano Test**: Compare forecast accuracy vs baselines
- **Block Bootstrap**: 95% confidence intervals (20-day blocks, 5000 resamples)
- **Cost Sensitivity**: Sharpe degradation across transaction cost regimes

---

## Visualizations

The pipeline generates 9 publication-quality figures:

1. **Equity Curves**: Comparison across all strategies (2022-2025)
2. **RL Equity**: Detailed view of RL agent performance
3. **Drawdown**: Underwater chart showing peak-to-trough declines
4. **Rolling Sharpe**: 63-day rolling risk-adjusted performance
5. **Weights Heatmap**: Time × asset allocation matrix
6. **Weights Area**: Stacked area chart (full investment constraint)
7. **Weight Statistics**: Distribution across assets (box plots)
8. **Turnover vs Sharpe**: Efficiency scatter plot
9. **Cost Sensitivity**: Performance degradation across cost regimes

---

## ⚙️ Configuration

All settings are in `config.yaml`:

```yaml
# Asset universe
assets: [SPY, QQQ, IWM, EFA, EEM, TLT, HYG, GLD, DBC]
exogenous: ["^VIX"]

# Data splits (temporal, no shuffling)
date:
  train: ["2012-01-01", "2018-12-31"]  # 7 years
  valid: ["2019-01-01", "2021-12-31"]  # 3 years
  test:  ["2022-01-01", "2025-10-31"]  # 4 years

# Trading constraints
trade:
  execute: "next_open"
  cost_bps_per_turnover: 25    # 25 bps roundtrip
  cap_per_asset: 0.30          # Max 30% per asset

# Reward function
reward:
  lambda_risk: 1.0             # Volatility penalty
  alpha_drawdown: 0.0          # Drawdown penalty (disabled)

# RL training
rl:
  algo: "PPO"
  total_timesteps: 800000
  eval_every_updates: 100      # Validation frequency
```

Modify these to experiment with different settings (e.g., increase `lambda_risk` for more conservative strategy).

---

## Research Extensions

Potential directions for future work:

**Quick Wins** (1-2 days):
- Walk-forward validation on multiple time periods
- Hyperparameter tuning (grid search over λ_risk, learning_rate)
- Expand asset universe (20+ ETFs, sector rotation)

**Medium Complexity** (1 week):
- Regime detection via Hidden Markov Models
- Multi-objective optimization (Pareto frontier for return/risk/turnover)
- Feature ablation studies (which features matter most?)

**Advanced** (2+ weeks):
- Online learning with experience replay and continual updates
- Transformer-based state encoder (replace hand-crafted features)
- Leverage and short-selling constraints
- Alternative execution models (market impact, slippage)

---

## Technical Report

For detailed methodology, results, and analysis, see **[RESEARCH_REPORT.pdf](RESEARCH_REPORT.pdf)**:
- Full problem formulation (MDP specification)
- Feature engineering details
- Algorithm descriptions (PPO objective, GAE)
- Statistical test methodology
- Interpretation and discussion
- References to prior work

---

## Troubleshooting

### Data Download Issues
- Ensure internet connectivity
- yfinance may occasionally fail; retry logic is built-in
- Some tickers may be delisted (warnings logged, not fatal)

### Training Performance
- Default settings require ~30-60 min on modern CPU
- Reduce `total_timesteps` in config.yaml for faster experiments
- GPU not required but can accelerate training

### Import Errors
```bash
pip install -r requirements.txt
```

Ensure Python 3.11+ is installed.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bindra2025deeprl,
  author = {Bindra, Jai Ansh},
  title = {Deep Reinforcement Learning for Multi-Asset Portfolio Rebalancing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JaiAnshSB26/deep-rl-rebalance}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Stable-Baselines3** team for excellent RL implementations
- **yfinance** for accessible market data
- **Gymnasium** for modern RL environment standards

---

## Contact

**Author**: Jai Ansh Bindra  
**GitHub**: [@JaiAnshSB26](https://github.com/JaiAnshSB26)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is a research project for educational purposes. Past performance does not guarantee future results. Not financial advice.
