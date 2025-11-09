"""
Download OHLCV data using yfinance and cache to parquet files.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_ohlcv(tickers: List[str], start: str, end: str, cache_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for given tickers and date range.
    
    Args:
        tickers: List of ticker symbols (e.g., ['SPY', 'QQQ', '^VIX'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        cache_dir: Directory to cache parquet files
        
    Returns:
        Dictionary mapping ticker to DataFrame with columns:
        [Date, Open, High, Low, Close, Adj Close, Volume]
    """
    import time
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    data = {}
    
    for ticker in tickers:
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        # Clean ticker name for file system (replace ^ with _)
        safe_ticker = ticker.replace('^', '_')
        parquet_file = cache_path / f"{safe_ticker}.parquet"
        
        # Try to load from cache first
        if parquet_file.exists():
            logger.info(f"Loading {ticker} from cache: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            
            # Check if cached data covers the requested range
            df_start = df.index.min()
            df_end = df.index.max()
            
            if df_start <= pd.Timestamp(start) and df_end >= pd.Timestamp(end):
                # Filter to requested range
                data[ticker] = df.loc[start:end].copy()
                continue
            else:
                logger.info(f"Cached data for {ticker} doesn't cover range, re-downloading")
        
        # Download from yfinance
        logger.info(f"Downloading {ticker} from {start} to {end}")
        
        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 2
        df = pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                # Use download method instead of Ticker.history for better reliability
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    threads=False  # Single-threaded to avoid rate limits
                )
                
                if df.empty:
                    if attempt < max_retries - 1:
                        logger.warning(f"No data for {ticker}, retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(f"No data returned for {ticker} after {max_retries} attempts")
                        df = pd.DataFrame()
                        break
                
                # Flatten multi-level columns if present (yf.download creates MultiIndex for single ticker)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Success - break retry loop
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error downloading {ticker}: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
                    df = pd.DataFrame()
                    break
        
        if df.empty:
            continue
        
        # Reset index to make Date a column, then set it back
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            # If yfinance already returned a DatetimeIndex, ensure it's tz-naive and sorted
            df.index = pd.to_datetime(df.index)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for {ticker}: {missing_cols}")
            # For some indices like VIX, we might only have Close
            if 'Close' in df.columns and ticker.startswith('^'):
                # Fill missing OHLC with Close for indices
                for col in ['Open', 'High', 'Low']:
                    if col not in df.columns:
                        df[col] = df['Close']
                if 'Adj Close' not in df.columns:
                    df['Adj Close'] = df['Close']
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
            else:
                logger.error(f"Cannot proceed with {ticker} due to missing columns")
                continue
        
        # Save to cache
        df.to_parquet(parquet_file)
        logger.info(f"Cached {ticker} to {parquet_file}")
        
        data[ticker] = df
    
    logger.info(f"Successfully loaded {len(data)} out of {len(tickers)} tickers")
    return data


if __name__ == "__main__":
    # Test download
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tickers = config['assets'] + config['exogenous']
    start_date = config['date']['train'][0]
    end_date = config['date']['test'][1]
    
    data = fetch_ohlcv(tickers, start_date, end_date)
    
    print("\nDownloaded data summary:")
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
