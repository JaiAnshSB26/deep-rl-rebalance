"""
Portfolio rebalancing environment for reinforcement learning.
Implements Gymnasium interface with cost-aware, risk-adjusted rewards.
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def project_to_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the probability simplex scaled by z.
    Uses efficient O(n log n) algorithm.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - z))[0][-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def project_to_capped_simplex(v: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the capped simplex: sum(w) = 1, w_i <= cap, w_i >= 0.
    Iterative clipping algorithm.
    """
    w = v.copy()
    max_iter = 100
    
    for _ in range(max_iter):
        # Clip to cap
        overflow = np.maximum(w - cap, 0).sum()
        w = np.minimum(w, cap)
        
        # Redistribute overflow to uncapped elements
        if overflow < 1e-10:
            break
        
        uncapped_mask = w < cap
        if uncapped_mask.sum() == 0:
            # All capped, redistribute equally
            w = np.full_like(w, cap)
            w = w / w.sum()  # Renormalize
            break
        
        w[uncapped_mask] += overflow / uncapped_mask.sum()
        
        # Project to simplex
        w = project_to_simplex(w, z=1.0)
    
    # Final normalization
    w = w / (w.sum() + 1e-10)
    
    return w


class PortfolioEnv(gym.Env):
    """
    Portfolio rebalancing environment.
    
    Observation: concatenation of:
        - Flattened features [N * F]
        - Current weights [N]
        - Rolling portfolio vol [1]
        - PCA factors of covariance [k]
    
    Action: Raw logits [N] -> softmax -> long-only weights with optional cap
    
    Reward: Cost-aware, risk-adjusted return with optional drawdown penalty
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        X: np.ndarray,  # [T, N, F] feature tensor
        R_next: np.ndarray,  # [T, N] next-period returns
        dates: np.ndarray,  # [T] date array
        cfg: dict,
        initial_weights: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        self.X = X
        self.R_next = R_next
        self.dates = dates
        self.cfg = cfg
        
        self.T, self.N, self.F = X.shape
        
        # Trading parameters
        self.cost_bps = cfg['trade']['cost_bps_per_turnover']
        self.cost_rate = self.cost_bps / 10000.0  # Convert bps to decimal
        self.cap = cfg['trade'].get('cap_per_asset', 1.0)
        
        # Reward parameters
        self.lambda_risk = cfg['reward']['lambda_risk']
        self.alpha_dd = cfg['reward']['alpha_drawdown']
        
        # Portfolio vol tracking
        self.vol_window = cfg['features'].get('portfolio_vol_window', 21)
        
        # PCA components (optional)
        self.k_pca = cfg['features'].get('cov_pca_components', 0)
        
        # State/action spaces
        obs_dim = self.N * self.F + self.N + 1 + self.k_pca
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: raw logits (will be passed through softmax)
        # Use finite bounds required by Stable-Baselines3
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.N,), dtype=np.float32
        )
        
        # Episode state
        self.t = 0
        self.weights = initial_weights if initial_weights is not None else np.ones(self.N) / self.N
        self.equity = 1.0
        self.peak = 1.0
        self.net_returns_history = []
        
        logger.info(f"PortfolioEnv initialized: T={self.T}, N={self.N}, F={self.F}, obs_dim={obs_dim}")
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Features (flattened)
        features_flat = self.X[self.t].flatten()  # [N * F]
        
        # Current weights
        weights = self.weights  # [N]
        
        # Rolling portfolio volatility
        if len(self.net_returns_history) >= 2:
            recent_rets = self.net_returns_history[-min(len(self.net_returns_history), self.vol_window):]
            portfolio_vol = np.std(recent_rets, ddof=1) if len(recent_rets) > 1 else 0.0
        else:
            portfolio_vol = 0.0
        
        # PCA of covariance (optional)
        if self.k_pca > 0:
            # Compute rolling covariance from recent returns
            # Simplified: use last 63 days of asset returns
            cov_window = self.cfg['features'].get('cov_window', 63)
            lookback = min(self.t, cov_window)
            
            if lookback >= 10:
                recent_asset_rets = self.R_next[max(0, self.t - lookback):self.t]  # [lookback, N]
                cov_matrix = np.cov(recent_asset_rets.T)
                
                # Compute top k eigenvalues
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                top_k_eigs = eigenvalues[-self.k_pca:][::-1]
                pca_factors = top_k_eigs / (np.sum(eigenvalues) + 1e-10)  # Normalized
            else:
                pca_factors = np.zeros(self.k_pca)
        else:
            pca_factors = np.array([])
        
        # Concatenate
        obs = np.concatenate([
            features_flat,
            weights,
            [portfolio_vol],
            pca_factors
        ]).astype(np.float32)
        
        return obs
    
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Convert action logits to valid portfolio weights."""
        # Softmax to ensure positivity and sum to 1
        exp_action = np.exp(action - np.max(action))  # Numerical stability
        weights = exp_action / exp_action.sum()
        
        # Apply cap if specified
        if self.cap < 1.0:
            weights = project_to_capped_simplex(weights, self.cap)
        
        return weights
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.t >= self.T - 1:
            # End of episode
            terminated = True
            truncated = False
            obs = self._get_obs()
            reward = 0.0
            info = {'episode_end': True}
            return obs, reward, terminated, truncated, info
        
        # Convert action to target weights
        target_weights = self._action_to_weights(action)
        
        # Compute turnover
        turnover = 0.5 * np.sum(np.abs(target_weights - self.weights))
        
        # Compute transaction cost
        cost = self.cost_rate * turnover
        
        # Get next period returns
        asset_returns = self.R_next[self.t]  # [N]
        
        # Gross portfolio return
        gross_return = np.dot(target_weights, asset_returns)
        
        # Net return (after costs)
        net_return = gross_return - cost
        
        # Update equity and peak for drawdown tracking
        self.equity *= (1.0 + net_return)
        prev_dd = (self.peak - (self.equity / (1.0 + net_return))) / (self.peak + 1e-10)
        self.peak = max(self.peak, self.equity)
        current_dd = (self.peak - self.equity) / (self.peak + 1e-10)
        dd_increment = current_dd - prev_dd
        
        # Store net return for rolling vol calculation
        self.net_returns_history.append(net_return)
        
        # Compute rolling portfolio vol
        if len(self.net_returns_history) >= 2:
            recent_rets = self.net_returns_history[-min(len(self.net_returns_history), self.vol_window):]
            portfolio_vol = np.std(recent_rets, ddof=1) if len(recent_rets) > 1 else 0.0
        else:
            portfolio_vol = 0.0
        
        # Reward: risk-adjusted, cost-aware
        reward = net_return - self.lambda_risk * portfolio_vol - self.alpha_dd * dd_increment
        
        # Update weights after market movement (pre-normalization)
        post_return_weights = target_weights * (1.0 + asset_returns)
        self.weights = post_return_weights / (post_return_weights.sum() + 1e-10)
        
        # Move to next time step
        self.t += 1
        
        # Get new observation
        obs = self._get_obs()
        
        # Check if episode is done
        terminated = (self.t >= self.T - 1)
        truncated = False
        
        # Info dict
        info = {
            'turnover': turnover,
            'cost': cost,
            'gross_return': gross_return,
            'net_return': net_return,
            'weights_before': target_weights.copy(),
            'weights_after': self.weights.copy(),
            'portfolio_vol': portfolio_vol,
            'equity': self.equity,
            'drawdown': current_dd,
            'date': self.dates[self.t - 1] if self.t > 0 else None
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.t = 0
        self.weights = np.ones(self.N) / self.N  # Equal weight initial
        self.equity = 1.0
        self.peak = 1.0
        self.net_returns_history = []
        
        obs = self._get_obs()
        info = {'reset': True}
        
        return obs, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass


def rollout_policy(
    policy,
    X: np.ndarray,
    R: np.ndarray,
    dates: np.ndarray,
    cfg: dict,
    deterministic: bool = True
) -> Dict:
    """
    Rollout a trained policy on given data (for evaluation).
    
    Args:
        policy: Trained SB3 policy
        X, R, dates: Environment data
        cfg: Configuration
        deterministic: Use deterministic policy
        
    Returns:
        Dictionary with daily_returns, weights, turnover, costs, info
    """
    env = PortfolioEnv(X, R, dates, cfg)
    obs, _ = env.reset()
    
    daily_returns = []
    weights_history = []
    turnover_history = []
    costs_history = []
    dates_history = []
    
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        
        if 'net_return' in info:
            daily_returns.append(info['net_return'])
            weights_history.append(info['weights_before'])
            turnover_history.append(info['turnover'])
            costs_history.append(info['cost'])
            if info.get('date') is not None:
                dates_history.append(info['date'])
    
    return {
        'daily_returns': np.array(daily_returns),
        'weights': np.array(weights_history),
        'turnover': np.array(turnover_history),
        'costs': np.array(costs_history),
        'dates': np.array(dates_history)
    }


if __name__ == "__main__":
    # Test environment
    import yaml
    from data.download import fetch_ohlcv
    from data.features import engineer_features
    from data.splits import make_splits, get_split_data
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Download and prepare data
    tickers = config['assets'] + config['exogenous']
    start_date = config['date']['train'][0]
    end_date = config['date']['test'][1]
    
    print("Downloading data...")
    data = fetch_ohlcv(tickers, start_date, end_date)
    
    asset_data = {k: v for k, v in data.items() if k in config['assets']}
    exog_data = {k: v for k, v in data.items() if k in config['exogenous']}
    
    print("Engineering features...")
    X, R = engineer_features(asset_data, exog_data, config)
    
    print("Creating splits...")
    dates = X.index.get_level_values('date').unique()
    splits = make_splits(dates, config)
    
    X_train, R_train, dates_train = get_split_data(X, R, splits['train'], config['assets'])
    
    print(f"Creating environment with shape X={X_train.shape}, R={R_train.shape}")
    env = PortfolioEnv(X_train, R_train, dates_train, config)
    
    # Test random actions
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.6f}, turnover={info['turnover']:.4f}, net_ret={info['net_return']:.6f}")
        
        if terminated or truncated:
            break
