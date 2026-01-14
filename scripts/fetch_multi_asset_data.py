"""
Multi-Asset Data Fetching Script for Graph-LORS Trader

Downloads and processes data for multiple assets used in dynamic correlation graph:
- Target asset: ^DJI (Dow Jones Industrial Average)
- Context assets: ^GSPC, ^VIX, GC=F, DX-Y.NYB

Strict anti-leakage protocol: All data aligned by trading dates.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from datetime import datetime

# Create data directory
os.makedirs('data', exist_ok=True)

# ============================================================================
# Asset Configuration
# ============================================================================
GRAPH_ASSETS = {
    'target': '^DJI',              # Target asset (Dow Jones)
    'context': [
        '^GSPC',                   # S&P 500 (US broad market)
        '^VIX',                    # Volatility index (risk sentiment)
        'GC=F',                    # Gold futures (safe haven)
        'DX-Y.NYB',                # USD index (macro proxy)
    ]
}

# All assets for fetching
ALL_ASSETS = [GRAPH_ASSETS['target']] + GRAPH_ASSETS['context']

# Date range (consistent with original project)
START_DATE = '2016-01-01'
END_DATE = '2025-01-01'


def fetch_single_asset(ticker, start_date, end_date):
    """
    Fetch data for a single asset

    Args:
        ticker: Asset ticker symbol
        start_date: Start date string
        end_date: End date string

    Returns:
        DataFrame with OHLCV data
    """
    print(f"  Fetching {ticker}...")

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            print(f"    ⚠️ No data for {ticker}")
            return None

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Keep only OHLCV columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data['Ticker'] = ticker

        print(f"    ✅ {len(data)} trading days fetched")
        return data

    except Exception as e:
        print(f"    ❌ Error fetching {ticker}: {e}")
        return None


def add_technical_indicators(df):
    """
    Add technical indicators (consistent with original project)

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicators
    """
    df = df.copy()

    # RSI (14-day)
    rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # MACD (使用 macd() 而非 macd_diff()，与原始 fetch_data.py 保持一致)
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()

    # Bollinger Bands (20-day)
    bb = BollingerBands(df['Close'], window=20)
    df['Bollinger_High'] = bb.bollinger_hband()
    df['Bollinger_Low'] = bb.bollinger_lband()

    # 20-day volatility (使用价格标准差，与原始 fetch_data.py 保持一致)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()

    # Drop NaN rows (from indicator calculations)
    df = df.dropna()

    return df


def compute_graph_node_features(df):
    """
    Compute node features for graph construction

    These are summary statistics used as node features in GAT.
    Computed over rolling windows to maintain temporal consistency.

    Args:
        df: DataFrame with OHLCV + indicators

    Returns:
        DataFrame with node features
    """
    df = df.copy()

    # Rolling statistics (60-day window, consistent with correlation window)
    window = 60

    # Return statistics
    returns = df['Close'].pct_change()
    df['Return_Mean_60'] = returns.rolling(window).mean()
    df['Return_Std_60'] = returns.rolling(window).std()
    df['Return_Skew_60'] = returns.rolling(window).skew()

    # Price momentum
    df['Momentum_20'] = df['Close'].pct_change(20)
    df['Momentum_60'] = df['Close'].pct_change(60)

    return df


def align_multi_asset_data(asset_data_dict):
    """
    Align multiple asset data by common trading dates

    Critical for anti-leakage: ensures all assets have same timestamps

    Args:
        asset_data_dict: Dict of {ticker: DataFrame}

    Returns:
        Dict of aligned DataFrames
    """
    print("\n" + "="*70)
    print("Aligning Multi-Asset Data")
    print("="*70)

    # Find common dates across all assets
    date_sets = [set(df.index) for df in asset_data_dict.values() if df is not None]

    if not date_sets:
        raise ValueError("No valid asset data to align")

    common_dates = date_sets[0]
    for ds in date_sets[1:]:
        common_dates = common_dates.intersection(ds)

    common_dates = sorted(list(common_dates))
    print(f"Common trading days: {len(common_dates)}")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]}")

    # Filter each asset to common dates
    aligned_data = {}
    for ticker, df in asset_data_dict.items():
        if df is not None:
            aligned_df = df.loc[common_dates].copy()
            aligned_data[ticker] = aligned_df
            print(f"  {ticker}: {len(aligned_df)} days")

    return aligned_data, common_dates


def create_price_matrix(aligned_data, common_dates):
    """
    Create price matrix for correlation computation

    Args:
        aligned_data: Dict of aligned DataFrames
        common_dates: List of common dates

    Returns:
        DataFrame with Close prices for all assets
    """
    price_matrix = pd.DataFrame(index=common_dates)

    for ticker, df in aligned_data.items():
        price_matrix[ticker] = df['Close']

    return price_matrix


def main():
    """Main function to fetch and process multi-asset data"""
    print("\n" + "="*70)
    print("Graph-LORS Trader: Multi-Asset Data Preparation")
    print("="*70)
    print(f"Target asset: {GRAPH_ASSETS['target']}")
    print(f"Context assets: {GRAPH_ASSETS['context']}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Step 1: Fetch all assets
    print("\n" + "="*70)
    print("Step 1: Fetching Asset Data")
    print("="*70)

    raw_data = {}
    for ticker in ALL_ASSETS:
        data = fetch_single_asset(ticker, START_DATE, END_DATE)
        if data is not None:
            raw_data[ticker] = data

    print(f"\nSuccessfully fetched: {len(raw_data)}/{len(ALL_ASSETS)} assets")

    # Step 2: Align data by common dates
    aligned_data, common_dates = align_multi_asset_data(raw_data)

    # Step 3: Add technical indicators to target asset
    print("\n" + "="*70)
    print("Step 3: Adding Technical Indicators (Target Asset)")
    print("="*70)

    target_ticker = GRAPH_ASSETS['target']
    target_data = aligned_data[target_ticker].copy()
    target_data = add_technical_indicators(target_data)

    # Update aligned data with indicators
    # Need to re-align after dropping NaN from indicators
    valid_dates = target_data.index

    for ticker in aligned_data:
        aligned_data[ticker] = aligned_data[ticker].loc[valid_dates]

    aligned_data[target_ticker] = target_data
    common_dates = list(valid_dates)

    print(f"Final data points: {len(common_dates)}")

    # Step 4: Compute node features for all assets
    print("\n" + "="*70)
    print("Step 4: Computing Graph Node Features")
    print("="*70)

    for ticker in aligned_data:
        aligned_data[ticker] = compute_graph_node_features(aligned_data[ticker])
        # Drop NaN from rolling computations
        aligned_data[ticker] = aligned_data[ticker].dropna()

    # Re-align after node feature computation
    date_sets = [set(df.index) for df in aligned_data.values()]
    final_dates = sorted(list(set.intersection(*date_sets)))

    for ticker in aligned_data:
        aligned_data[ticker] = aligned_data[ticker].loc[final_dates]

    print(f"Final aligned data points: {len(final_dates)}")

    # Step 5: Save processed data
    print("\n" + "="*70)
    print("Step 5: Saving Processed Data")
    print("="*70)

    # Save target asset data (使用新文件名，避免覆盖原有基线数据)
    # 原有 data/DJI_2016_2025.csv 保留用于基线模型对比
    target_save_path = 'data/DJI_graph_2016_2025.csv'
    target_df = aligned_data[target_ticker].copy()
    target_df['Ticker'] = target_ticker
    target_df.to_csv(target_save_path)
    print(f"✅ Target asset (graph-aligned) saved: {target_save_path}")
    print(f"   ℹ️  Original baseline data preserved: data/DJI_2016_2025.csv")

    # Save all assets combined (for graph construction)
    all_assets_df = pd.concat(aligned_data.values(), keys=aligned_data.keys())
    all_assets_path = 'data/multi_asset_2016_2025.csv'
    all_assets_df.to_csv(all_assets_path)
    print(f"✅ Multi-asset data saved: {all_assets_path}")

    # Save price matrix (for correlation computation)
    price_matrix = create_price_matrix(aligned_data, final_dates)
    price_matrix_path = 'data/price_matrix_2016_2025.csv'
    price_matrix.to_csv(price_matrix_path)
    print(f"✅ Price matrix saved: {price_matrix_path}")

    # Save asset configuration
    import json
    config_path = 'data/graph_assets_config.json'
    config = {
        'target': GRAPH_ASSETS['target'],
        'context': GRAPH_ASSETS['context'],
        'all_assets': ALL_ASSETS,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'n_trading_days': len(final_dates),
        'date_range': [str(final_dates[0]), str(final_dates[-1])],
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Asset config saved: {config_path}")

    # Summary
    print("\n" + "="*70)
    print("Data Preparation Complete")
    print("="*70)
    print(f"Total assets: {len(ALL_ASSETS)}")
    print(f"Trading days: {len(final_dates)}")
    print(f"Date range: {final_dates[0]} to {final_dates[-1]}")
    print("\nFeatures per asset:")
    print(f"  OHLCV: Open, High, Low, Close, Volume")
    print(f"  Indicators: RSI, MACD, Bollinger_High, Bollinger_Low, Volatility_20")
    print(f"  Node features: Return_Mean_60, Return_Std_60, Return_Skew_60, Momentum_20, Momentum_60")

    return aligned_data, final_dates


if __name__ == "__main__":
    main()
