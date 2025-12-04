"""
Data Fetching Script

Downloads financial data from Yahoo Finance and processes it:
- Downloads OHLCV data for multiple tickers
- Computes technical indicators (RSI, MACD, Bollinger Bands, Volatility)
- Saves processed data to CSV

Usage:
    python scripts/fetch_data.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os

SEED = 1234
np.random.seed(SEED)

# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

START_DATE = '2016-01-01'
END_DATE = '2025-01-01'

# Define ticker list
tickers = [
    '^DJI', '^GSPC', '^IXIC',  # US indices
    'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU',  # Sector ETFs
    '^FTSE', '^N225', '^HSI', '^BSESN', '^BVSP',  # International indices
    'GC=F', 'SI=F', 'CL=F',  # Commodities
    'EURUSD=X',  # Forex
    'BTC-USD'  # Crypto
]


def fetch_data(tickers, start=START_DATE, end=END_DATE):
    """
    Download data for specified tickers

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        Downloaded data
    """
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    return data


def process_ticker_data(ticker, data):
    """
    Process individual ticker data to generate 10 features

    Features:
    - OHLCV (5 features)
    - Technical indicators (5 features): RSI, MACD, Bollinger Bands (High/Low), Volatility

    Args:
        ticker: Ticker symbol
        data: Downloaded data

    Returns:
        Processed DataFrame with 10 features
    """
    ticker_data = data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Compute technical indicators
    ticker_data['RSI'] = ta.momentum.RSIIndicator(ticker_data['Close']).rsi()
    ticker_data['MACD'] = ta.trend.MACD(ticker_data['Close']).macd()

    bb = ta.volatility.BollingerBands(ticker_data['Close'])
    ticker_data['Bollinger_High'] = bb.bollinger_hband()
    ticker_data['Bollinger_Low'] = bb.bollinger_lband()

    ticker_data['Volatility_20'] = ticker_data['Close'].rolling(20).std()

    # Select 10 features
    ticker_data = ticker_data[[
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low', 'Volatility_20'
    ]]

    # Handle missing values
    ticker_data = ticker_data.ffill().bfill()
    ticker_data['Ticker'] = ticker
    return ticker_data


def main():
    """Main execution function"""
    print("="*70)
    print("Fetching Financial Data")
    print("="*70)
    print(f"Tickers: {len(tickers)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Fetch and process data
    data = fetch_data(tickers)
    all_data = pd.concat([process_ticker_data(ticker, data) for ticker in tickers], axis=0)

    # Save to CSV
    output_file = 'data/all_tickers.csv'
    all_data.to_csv(output_file)

    print(f"\n✅ Saved: {output_file}")
    print(f"Total rows: {len(all_data)}")
    print(f"Columns: {list(all_data.columns)}")
    print("="*70)


if __name__ == "__main__":
    main()
