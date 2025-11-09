"""
PPO training harness with validation-based model selection.
"""
import os
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationCallback(BaseCallback):
    """
    Callback for periodic validation and model checkpointing.
    """
    
    def __init__(
        self,
        eval_env_fn: Callable,
        eval_freq: int,
        log_dir: str,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_sharpe = -np.inf
        self.best_model_path = None
        self.eval_count = 0
        self.plateau_count = 0
        self.sharpe_history = []
    
    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            # Run validation
            logger.info(f"\nValidation #{self.eval_count} at step {self.n_calls}")
            results = self.eval_env_fn(self.model)
            
            # Compute Sharpe ratio
            daily_rets = results['daily_returns']
            if len(daily_rets) > 0:
                mean_ret = np.mean(daily_rets)
                std_ret = np.std(daily_rets, ddof=1)
                sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252)
            else:
                sharpe = -np.inf
            
            self.sharpe_history.append(sharpe)
            
            logger.info(f"Validation Sharpe: {sharpe:.4f}")
            
            # Check if best
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.plateau_count = 0
                
                # Save model
                model_path = self.log_dir / f"best_model_sharpe_{sharpe:.4f}.zip"
                self.model.save(model_path)
                
                # Save normalized env if available
                if hasattr(self.model, 'get_vec_normalize_env'):
                    vec_norm = self.model.get_vec_normalize_env()
                    if vec_norm is not None:
                        vec_norm.save(self.log_dir / "vec_normalize.pkl")
                
                self.best_model_path = str(model_path)
                logger.info(f"New best model saved: {model_path}")
            else:
                self.plateau_count += 1
                logger.info(f"Plateau count: {self.plateau_count}")
            
            # Early stopping check (optional)
            if self.plateau_count >= 10:
                logger.warning("Validation Sharpe plateaued for 10 evaluations, consider stopping")
        
        return True  # Continue training


def train_ppo(
    env_train,
    env_valid_eval_fn: Callable,
    cfg: dict,
    log_dir: str = "results/logs/ppo",
    normalize_obs: bool = True
) -> str:
    """
    Train PPO agent with validation-based model selection.
    
    Args:
        env_train: Training environment
        env_valid_eval_fn: Function that takes a policy and returns validation results
        cfg: Configuration dictionary
        log_dir: Directory for logs and checkpoints
        normalize_obs: Whether to normalize observations
        
    Returns:
        Path to best checkpoint
    """
    ppo_cfg = cfg['rl']['ppo']
    reward_scale = cfg['rl'].get('reward_scale', 100.0)
    
    logger.info("Setting up PPO training")
    logger.info(f"Config: {ppo_cfg}")
    
    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env_train])
    
    # Normalize observations (not rewards, we scale manually)
    if normalize_obs:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,  # We handle reward scaling ourselves
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Import torch for activation function
    import torch

    # Check for TensorBoard availability. stable-baselines3 will raise an ImportError
    # if tensorboard is not installed but tensorboard_log is provided, so only enable
    # tensorboard logging when the SummaryWriter is importable.
    try:
        # The SummaryWriter is provided by torch.utils.tensorboard when installed
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        tb_available = True
    except Exception:
        tb_available = False

    if not tb_available:
        logger.warning(
            "TensorBoard not found. Disabling tensorboard logging. "
            "Install it with: python -m pip install tensorboard"
        )

    tb_log = log_dir if tb_available else None

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        learning_rate=ppo_cfg['learning_rate'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        clip_range=ppo_cfg['clip_range'],
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh
        ),
        verbose=1,
        tensorboard_log=tb_log  # log_dir
    )
    
    logger.info(f"PPO model created with {model.policy}")
    
    # Setup validation callback
    eval_freq = cfg['rl']['eval_every_updates'] * ppo_cfg['n_steps']
    callback = ValidationCallback(
        eval_env_fn=env_valid_eval_fn,
        eval_freq=eval_freq,
        log_dir=log_dir
    )
    
    # Train
    total_timesteps = cfg['rl']['total_timesteps']
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    logger.info(f"Training complete. Best Sharpe: {callback.best_sharpe:.4f}")
    logger.info(f"Best model: {callback.best_model_path}")
    
    return callback.best_model_path


def load_ppo_model(model_path: str, vec_normalize_path: str = None):
    """Load trained PPO model."""
    import torch  # Import here to avoid issues if not installed
    
    model = PPO.load(model_path)
    
    # Load VecNormalize if exists
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_norm = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: None]))
        vec_norm.training = False
        vec_norm.norm_reward = False
        return model, vec_norm
    
    return model, None


if __name__ == "__main__":
    # Test PPO training
    import yaml
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.download import fetch_ohlcv
    from data.features import engineer_features
    from data.splits import make_splits, get_split_data
    from envs.portfolio_env import PortfolioEnv, rollout_policy
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    np.random.seed(config['seed'])
    import torch
    torch.manual_seed(config['seed'])
    
    # Download and prepare data
    print("Downloading data...")
    tickers = config['assets'] + config['exogenous']
    start_date = config['date']['train'][0]
    end_date = config['date']['test'][1]
    
    data = fetch_ohlcv(tickers, start_date, end_date)
    asset_data = {k: v for k, v in data.items() if k in config['assets']}
    exog_data = {k: v for k, v in data.items() if k in config['exogenous']}
    
    print("Engineering features...")
    X, R = engineer_features(asset_data, exog_data, config)
    
    print("Creating splits...")
    dates = X.index.get_level_values('date').unique()
    splits = make_splits(dates, config)
    
    X_train, R_train, dates_train = get_split_data(X, R, splits['train'], config['assets'])
    X_valid, R_valid, dates_valid = get_split_data(X, R, splits['valid'], config['assets'])
    
    # Create environments
    env_train = PortfolioEnv(X_train, R_train, dates_train, config)
    
    def eval_fn(policy):
        return rollout_policy(policy, X_valid, R_valid, dates_valid, config)
    
    # Train (with reduced timesteps for testing)
    config_test = config.copy()
    config_test['rl']['total_timesteps'] = 10000
    
    best_model = train_ppo(env_train, eval_fn, config_test)
    print(f"Best model saved to: {best_model}")
