"""
Create train/validation/test splits by date ranges.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_splits(dates: pd.Index, cfg: dict, save_path: str = "data/splits.json") -> Dict[str, slice]:
    """
    Create train/valid/test splits based on date ranges.
    
    Args:
        dates: DatetimeIndex with all available dates
        cfg: Configuration dictionary with 'date' section
        save_path: Path to save split information
        
    Returns:
        Dictionary with 'train', 'valid', 'test' slices (integer-based)
    """
    date_cfg = cfg['date']
    
    train_start, train_end = date_cfg['train']
    valid_start, valid_end = date_cfg['valid']
    test_start, test_end = date_cfg['test']
    
    # Convert to datetime
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    
    # Ensure dates is a DatetimeIndex
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    
    # Find integer indices
    train_mask = (dates >= train_start) & (dates <= train_end)
    valid_mask = (dates >= valid_start) & (dates <= valid_end)
    test_mask = (dates >= test_start) & (dates <= test_end)
    
    train_indices = np.where(train_mask)[0]
    valid_indices = np.where(valid_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    if len(train_indices) == 0:
        raise ValueError(f"No training data found between {train_start} and {train_end}")
    if len(valid_indices) == 0:
        raise ValueError(f"No validation data found between {valid_start} and {valid_end}")
    if len(test_indices) == 0:
        raise ValueError(f"No test data found between {test_start} and {test_end}")
    
    # Create slices
    splits = {
        'train': slice(train_indices[0], train_indices[-1] + 1),
        'valid': slice(valid_indices[0], valid_indices[-1] + 1),
        'test': slice(test_indices[0], test_indices[-1] + 1)
    }
    
    # Save split information
    split_info = {
        'train': {
            'start': train_start.strftime('%Y-%m-%d'),
            'end': train_end.strftime('%Y-%m-%d'),
            'start_idx': int(train_indices[0]),
            'end_idx': int(train_indices[-1]),
            'length': int(len(train_indices)),
            'actual_start': dates[train_indices[0]].strftime('%Y-%m-%d'),
            'actual_end': dates[train_indices[-1]].strftime('%Y-%m-%d')
        },
        'valid': {
            'start': valid_start.strftime('%Y-%m-%d'),
            'end': valid_end.strftime('%Y-%m-%d'),
            'start_idx': int(valid_indices[0]),
            'end_idx': int(valid_indices[-1]),
            'length': int(len(valid_indices)),
            'actual_start': dates[valid_indices[0]].strftime('%Y-%m-%d'),
            'actual_end': dates[valid_indices[-1]].strftime('%Y-%m-%d')
        },
        'test': {
            'start': test_start.strftime('%Y-%m-%d'),
            'end': test_end.strftime('%Y-%m-%d'),
            'start_idx': int(test_indices[0]),
            'end_idx': int(test_indices[-1]),
            'length': int(len(test_indices)),
            'actual_start': dates[test_indices[0]].strftime('%Y-%m-%d'),
            'actual_end': dates[test_indices[-1]].strftime('%Y-%m-%d')
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"Splits created and saved to {save_path}")
    logger.info(f"Train: {split_info['train']['length']} days ({split_info['train']['actual_start']} to {split_info['train']['actual_end']})")
    logger.info(f"Valid: {split_info['valid']['length']} days ({split_info['valid']['actual_start']} to {split_info['valid']['actual_end']})")
    logger.info(f"Test: {split_info['test']['length']} days ({split_info['test']['actual_start']} to {split_info['test']['actual_end']})")
    
    return splits


def get_split_data(X: pd.DataFrame, R: pd.DataFrame, split: slice, assets: list) -> tuple:
    """
    Extract data for a specific split.
    
    Args:
        X: MultiIndex DataFrame (date, asset) with features
        R: DataFrame with date index and asset columns
        split: slice object for the split
        assets: List of asset tickers
        
    Returns:
        X_split: [T, N, F] numpy array
        R_split: [T, N] numpy array
        dates_split: [T] numpy array of dates
    """
    # Get unique dates from X
    dates = X.index.get_level_values('date').unique().sort_values()
    dates_split = dates[split]
    
    # Filter R by dates
    R_split = R.loc[dates_split].values  # [T, N]
    
    # Filter X by dates and reshape
    X_filtered = X.loc[dates_split]
    
    # Reshape X to [T, N, F]
    T = len(dates_split)
    N = len(assets)
    F = X.shape[1]
    
    X_split = np.zeros((T, N, F))
    
    for i, date in enumerate(dates_split):
        for j, asset in enumerate(assets):
            if (date, asset) in X_filtered.index:
                X_split[i, j, :] = X_filtered.loc[(date, asset)].values
    
    return X_split, R_split, dates_split.values


if __name__ == "__main__":
    # Test splits
    import yaml
    from download import fetch_ohlcv
    from features import engineer_features
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Download and engineer features
    tickers = config['assets'] + config['exogenous']
    start_date = config['date']['train'][0]
    end_date = config['date']['test'][1]
    
    data = fetch_ohlcv(tickers, start_date, end_date)
    asset_data = {k: v for k, v in data.items() if k in config['assets']}
    exog_data = {k: v for k, v in data.items() if k in config['exogenous']}
    
    X, R = engineer_features(asset_data, exog_data, config)
    
    # Create splits
    dates = X.index.get_level_values('date').unique()
    splits = make_splits(dates, config)
    
    # Test data extraction
    X_train, R_train, dates_train = get_split_data(X, R, splits['train'], config['assets'])
    X_valid, R_valid, dates_valid = get_split_data(X, R, splits['valid'], config['assets'])
    X_test, R_test, dates_test = get_split_data(X, R, splits['test'], config['assets'])
    
    print(f"\nTrain split shape: X={X_train.shape}, R={R_train.shape}")
    print(f"Valid split shape: X={X_valid.shape}, R={R_valid.shape}")
    print(f"Test split shape: X={X_test.shape}, R={R_test.shape}")
