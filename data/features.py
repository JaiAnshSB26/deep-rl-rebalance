"""
Feature engineering for portfolio environment.
All features must be causal (known at decision time t using data up to t-1).
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.stats import mstats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_returns(prices: pd.Series, lags: list) -> pd.DataFrame:
    """Compute lagged log returns."""
    ret_df = pd.DataFrame(index=prices.index)
    for lag in lags:
        ret_df[f'ret_lag{lag}'] = np.log(prices / prices.shift(lag))
    return ret_df


def compute_rolling_stats(returns: pd.Series, windows: list, stat: str) -> pd.DataFrame:
    """Compute rolling statistics (mean or std)."""
    stat_df = pd.DataFrame(index=returns.index)
    for window in windows:
        if stat == 'mean':
            stat_df[f'roll_mean_{window}'] = returns.rolling(window, min_periods=max(1, window // 2)).mean()
        elif stat == 'std':
            stat_df[f'roll_std_{window}'] = returns.rolling(window, min_periods=max(1, window // 2)).std(ddof=1)
    return stat_df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=max(1, period // 2)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=max(1, period // 2)).mean()
    
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': macd_hist
    }, index=prices.index)


def compute_bollinger(prices: pd.Series, period: int = 20) -> pd.Series:
    """Compute Bollinger %B indicator."""
    sma = prices.rolling(window=period, min_periods=max(1, period // 2)).mean()
    std = prices.rolling(window=period, min_periods=max(1, period // 2)).std(ddof=1)
    
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    
    # %B = (Price - Lower Band) / (Upper Band - Lower Band)
    bb_pct = (prices - lower_band) / (upper_band - lower_band).replace(0, 1e-10)
    return bb_pct


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=max(1, period // 2)).mean()
    return atr


def winsorize_cross_section(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize features by cross-section (per date)."""
    return df.apply(lambda row: mstats.winsorize(row, limits=[lower, 1 - upper]), axis=1, result_type='broadcast')


def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize across assets per date."""
    return df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-10), axis=1)


def engineer_features(
    prices: Dict[str, pd.DataFrame],
    exog: Dict[str, pd.DataFrame],
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer causal features for all assets and compute next-period returns.
    
    Args:
        prices: Dict mapping ticker to OHLCV DataFrame
        exog: Dict mapping exogenous ticker (e.g., ^VIX) to DataFrame
        config: Configuration dictionary
        
    Returns:
        X: MultiIndex DataFrame (date, asset) with features (winsorized + z-scored per day)
        R: DataFrame with date index and next-period returns for each asset
    """
    cfg = config['features']
    trade_cfg = config['trade']
    assets = config['assets']
    
    # Determine execution convention for returns
    execute = trade_cfg['execute']
    
    logger.info(f"Engineering features for {len(assets)} assets with execution: {execute}")
    
    # Build feature dataframe for each asset
    asset_features = {}
    next_returns = {}
    
    for ticker in assets:
        if ticker not in prices:
            logger.warning(f"Skipping {ticker}: not in price data")
            continue
        
        df = prices[ticker].copy()
        
        # Use Adjusted Close for calculations
        price = df['Adj Close']
        
        # --- Compute daily returns for rolling stats ---
        daily_ret = np.log(price / price.shift(1))
        
        # --- Lagged returns ---
        ret_features = compute_returns(price, cfg['lags'])
        
        # --- Rolling statistics ---
        roll_mean = compute_rolling_stats(daily_ret, cfg['roll_mean'], 'mean')
        roll_std = compute_rolling_stats(daily_ret, cfg['roll_std'], 'std')
        
        # --- Technical indicators ---
        rsi = compute_rsi(price, cfg['rsi'])
        macd_df = compute_macd(price, *cfg['macd'])
        bb = compute_bollinger(price, cfg['bb'])
        
        # ATR (optional, needs high/low)
        if 'High' in df.columns and 'Low' in df.columns:
            atr = compute_atr(df['High'], df['Low'], df['Close'], cfg.get('atr', 14))
            atr_norm = atr / price  # Normalize by price
        else:
            atr_norm = pd.Series(0, index=price.index)
        
        # --- Combine all features ---
        features = pd.concat([
            ret_features,
            roll_mean,
            roll_std,
            pd.DataFrame({'rsi': rsi}),
            macd_df,
            pd.DataFrame({'bb_pct': bb}),
            pd.DataFrame({'atr_norm': atr_norm})
        ], axis=1)
        
        asset_features[ticker] = features
        
        # --- Compute next-period returns based on execution convention ---
        if execute == 'next_open':
            # Execute at next day's open
            # Return from t+1 open to t+2 open (or t+1 close as proxy)
            # Simplified: use close-to-close shifted by 1
            next_ret = np.log(price.shift(-1) / price)
        elif execute == 'next_close':
            # Execute at next day's close
            # Return from t+1 close to t+2 close
            next_ret = np.log(price.shift(-2) / price.shift(-1))
        else:
            raise ValueError(f"Unknown execution convention: {execute}")
        
        next_returns[ticker] = next_ret
    
    # --- Add exogenous/market features (shared across all assets) ---
    exog_features = {}
    
    for exog_ticker in config['exogenous']:
        if exog_ticker not in exog:
            logger.warning(f"Exogenous ticker {exog_ticker} not found")
            continue
        
        exog_df = exog[exog_ticker]
        exog_price = exog_df['Adj Close'] if 'Adj Close' in exog_df.columns else exog_df['Close']
        
        # Lagged level and 5-day change
        exog_features[f'{exog_ticker}_level'] = exog_price.shift(1)  # Causal: use previous day
        exog_features[f'{exog_ticker}_chg5'] = exog_price.shift(1) - exog_price.shift(6)
    
    # Market return (SPY lagged 1-day return)
    if 'SPY' in prices:
        spy_price = prices['SPY']['Adj Close']
        spy_ret = np.log(spy_price / spy_price.shift(1)).shift(1)  # Lagged
        exog_features['market_ret_lag1'] = spy_ret
    
    exog_df = pd.DataFrame(exog_features)
    
    # --- Create MultiIndex DataFrame ---
    # Stack all asset features
    feature_list = []
    for ticker in assets:
        if ticker not in asset_features:
            continue
        feat = asset_features[ticker].copy()
        feat['asset'] = ticker
        feat['date'] = feat.index
        feature_list.append(feat)
    
    if not feature_list:
        raise ValueError("No features were generated for any asset")
    
    # Concatenate
    all_features = pd.concat(feature_list, axis=0, ignore_index=True)
    all_features = all_features.set_index(['date', 'asset'])
    
    # --- Merge exogenous features (broadcast to all assets) ---
    # Create a temporary dataframe to merge
    exog_expanded = []
    for ticker in assets:
        if ticker not in asset_features:
            continue
        exog_temp = exog_df.copy()
        exog_temp['asset'] = ticker
        exog_temp['date'] = exog_temp.index
        exog_expanded.append(exog_temp)
    
    if exog_expanded:
        exog_multi = pd.concat(exog_expanded, axis=0, ignore_index=True)
        exog_multi = exog_multi.set_index(['date', 'asset'])
        all_features = all_features.join(exog_multi, how='left')
    
    # --- Cross-sectional normalization (per date) ---
    logger.info("Applying cross-sectional winsorization and z-scoring")
    
    # Group by date and apply transformations
    def normalize_date_group(group):
        # Winsorize
        winsorized = winsorize_cross_section(group)
        # Z-score
        zscored = zscore_cross_section(winsorized)
        return zscored
    
    # Apply only to non-categorical columns
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns
    
    X_normalized = all_features.copy()
    
    # Apply normalization date by date to avoid MultiIndex issues
    for date in all_features.index.get_level_values('date').unique():
        date_mask = all_features.index.get_level_values('date') == date
        date_data = all_features.loc[date_mask, numeric_cols]
        normalized_date = normalize_date_group(date_data)
        # Assign directly back to the same positions
        X_normalized.loc[date_mask, numeric_cols] = normalized_date.values
    
    # Forward fill any remaining NaNs from rolling windows
    X_normalized = X_normalized.fillna(method='ffill').fillna(0)
    
    # --- Prepare return matrix R ---
    ret_df = pd.DataFrame(next_returns)
    ret_df = ret_df[assets]  # Ensure consistent ordering
    
    logger.info(f"Feature engineering complete: {X_normalized.shape[0]} observations, {X_normalized.shape[1]} features")
    logger.info(f"Return matrix shape: {ret_df.shape}")
    
    return X_normalized, ret_df


if __name__ == "__main__":
    # Test feature engineering
    import yaml
    from download import fetch_ohlcv
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Download data
    tickers = config['assets'] + config['exogenous']
    start_date = config['date']['train'][0]
    end_date = config['date']['test'][1]
    
    data = fetch_ohlcv(tickers, start_date, end_date)
    
    # Split into assets and exogenous
    asset_data = {k: v for k, v in data.items() if k in config['assets']}
    exog_data = {k: v for k, v in data.items() if k in config['exogenous']}
    
    # Engineer features
    X, R = engineer_features(asset_data, exog_data, config)
    
    print("\nFeature matrix (first 5 rows):")
    print(X.head())
    print("\nReturn matrix (first 5 rows):")
    print(R.head())
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Number of features: {len(X.columns)}")
