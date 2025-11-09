"""
SAC training harness (secondary algorithm).
"""
import os
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationCallback(BaseCallback):
    """Callback for periodic validation and model checkpointing."""
    
    def __init__(self, eval_env_fn: Callable, eval_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_sharpe = -np.inf
        self.best_model_path = None
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            logger.info(f"\nValidation #{self.eval_count} at step {self.n_calls}")
            results = self.eval_env_fn(self.model)
            
            daily_rets = results['daily_returns']
            if len(daily_rets) > 0:
                sharpe = (np.mean(daily_rets) / (np.std(daily_rets, ddof=1) + 1e-10)) * np.sqrt(252)
            else:
                sharpe = -np.inf
            
            logger.info(f"Validation Sharpe: {sharpe:.4f}")
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                model_path = self.log_dir / f"best_model_sharpe_{sharpe:.4f}.zip"
                self.model.save(model_path)
                self.best_model_path = str(model_path)
                logger.info(f"New best model saved: {model_path}")
        
        return True


def train_sac(
    env_train,
    env_valid_eval_fn: Callable,
    cfg: dict,
    log_dir: str = "results/logs/sac",
    normalize_obs: bool = True
) -> str:
    """Train SAC agent with validation-based model selection."""
    import torch
    
    sac_cfg = cfg['rl']['sac']
    
    logger.info("Setting up SAC training")
    logger.info(f"Config: {sac_cfg}")
    
    vec_env = DummyVecEnv([lambda: env_train])
    
    if normalize_obs:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=sac_cfg['learning_rate'],
        buffer_size=sac_cfg['buffer_size'],
        batch_size=sac_cfg['batch_size'],
        gamma=sac_cfg['gamma'],
        tau=sac_cfg['tau'],
        train_freq=sac_cfg['train_freq'],
        gradient_steps=sac_cfg['gradient_steps'],
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    eval_freq = cfg['rl']['eval_every_updates'] * 1000
    callback = ValidationCallback(eval_env_fn=env_valid_eval_fn, eval_freq=eval_freq, log_dir=log_dir)
    
    total_timesteps = cfg['rl']['total_timesteps']
    logger.info(f"Starting SAC training for {total_timesteps} timesteps")
    
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    logger.info(f"Training complete. Best Sharpe: {callback.best_sharpe:.4f}")
    return callback.best_model_path
