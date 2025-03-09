import pandas as pd
import numpy as np
import yfinance as yf

def fetch_yfinance_data(tickers, start_date, end_date):
    """
    fetches adjusted closing price data from Yahoo Finance.

    parameters:
    -----------
    tickers : list
        list of stock tickers.
    start_date : str
        start date in 'YYYY-MM-DD' format.
    end_date : str
        end date in 'YYYY-MM-DD' format.

    returns:
    --------
    prices : DataFrame
        adjusted closing prices of the given tickers.
    returns : DataFrame
        daily log returns of the given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)

    if data.empty or 'Close' not in data:
        raise ValueError("No valid stock data retrieved. Check tickers or API availability.")

    prices = data['Close']

    missing_tickers = [t for t in tickers if t not in prices.columns]
    if missing_tickers:
        print(f"Warning: The following tickers were not found and will be ignored: {missing_tickers}")

    returns = np.log(prices / prices.shift(1))

    return prices, returns

def prepare_data(tickers, benchmark, start_date, end_date, ff_url):
    """
    fetch financial data from Yahoo Finance and Fama-French factors.
    """
    prices, returns = fetch_yfinance_data(tickers + [benchmark], start_date, end_date)
    benchmark_prices = prices[benchmark]
    benchmark_returns = benchmark_prices.pct_change()

    return prices, returns, benchmark_prices, benchmark_returns
