"""
Quick test script to verify installation and basic functionality.
"""
import sys
import numpy as np

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    errors = []
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        errors.append(f"pandas: {e}")
    
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError as e:
        errors.append(f"yaml: {e}")
    
    try:
        import yfinance as yf
        print("  ✓ yfinance")
    except ImportError as e:
        errors.append(f"yfinance: {e}")
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError as e:
        errors.append(f"scipy: {e}")
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError as e:
        errors.append(f"scikit-learn: {e}")
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError as e:
        errors.append(f"matplotlib: {e}")
    
    try:
        import gymnasium
        print("  ✓ gymnasium")
    except ImportError as e:
        errors.append(f"gymnasium: {e}")
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import stable_baselines3
        print("  ✓ stable-baselines3")
    except ImportError as e:
        errors.append(f"stable-baselines3: {e}")
    
    if errors:
        print("\n❌ Missing dependencies:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def test_config():
    """Test config loading."""
    print("\nTesting config...")
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Config loaded with {len(config['assets'])} assets")
        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_modules():
    """Test custom modules."""
    print("\nTesting custom modules...")
    
    try:
        from data import download, features, splits
        print("  ✓ data modules")
    except ImportError as e:
        print(f"  ❌ data modules: {e}")
        return False
    
    try:
        from envs import portfolio_env
        print("  ✓ envs module")
    except ImportError as e:
        print(f"  ❌ envs module: {e}")
        return False
    
    try:
        from agents import ppo_trainer, sac_trainer
        print("  ✓ agents modules")
    except ImportError as e:
        print(f"  ❌ agents modules: {e}")
        return False
    
    try:
        from baselines import equal_weights, periodic_rebalance, risk_parity, momentum_tilt, mean_variance
        print("  ✓ baselines modules")
    except ImportError as e:
        print(f"  ❌ baselines modules: {e}")
        return False
    
    try:
        from metrics import evaluate, tests, utils
        print("  ✓ metrics modules")
    except ImportError as e:
        print(f"  ❌ metrics modules: {e}")
        return False
    
    try:
        from plots import equity, rolling, weights, sensitivity
        print("  ✓ plots modules")
    except ImportError as e:
        print(f"  ❌ plots modules: {e}")
        return False
    
    print("  ✅ All custom modules loadable!")
    return True


def main():
    print("="*80)
    print("DEEP RL PORTFOLIO REBALANCING - INSTALLATION TEST")
    print("="*80)
    
    all_ok = True
    
    # Test imports
    if not test_imports():
        all_ok = False
    
    # Test config
    if not test_config():
        all_ok = False
    
    # Test modules
    if not test_modules():
        all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou're ready to go! Next steps:")
        print("  1. Download data: python data/download.py")
        print("  2. Run full pipeline: jupyter notebook notebooks/01_train_evaluate.ipynb")
        print("  3. Or train PPO directly: python agents/ppo_trainer.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the errors above and try again.")
        print("Common fix: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
