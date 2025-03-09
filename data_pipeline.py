import pandas as pd
import numpy as np
import yfinance as yf

def fetch_yfinance_data(tickers, start_date, end_date):
    """
    Fetches adjusted closing price data from Yahoo Finance.

    Parameters:
    -----------
    tickers : list
        List of stock tickers.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns:
    --------
    prices : DataFrame
        Adjusted closing prices of the given tickers.
    returns : DataFrame
        Daily log returns of the given tickers.
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

def process_fama_french_factors(filepath):
    """
    Load and process the Fama-French factor dataset.

    Parameters:
    -----------
    filepath : str
        Path to the Fama-French CSV file.

    Returns:
    --------
    df : pd.DataFrame
        Processed DataFrame with cleaned date index.
    """
    try:
        df = pd.read_csv(filepath, skiprows=3)
        df = df[df.iloc[:, 0].str.match(r'^\d{8}$', na=False)]
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        if 'RF' not in df.columns:
            print("Warning: 'RF' column (risk-free rate) not found in Fama-French dataset.")

        return df

    except Exception as e:
        raise ValueError(f"Error processing Fama-French factors: {str(e)}")

def prepare_data(tickers, benchmark, start_date, end_date, ff_url):
    """
    Fetch financial data from Yahoo Finance and Fama-French factors.
    """
    prices, returns = fetch_yfinance_data(tickers + [benchmark], start_date, end_date)
    benchmark_prices = prices[benchmark]
    benchmark_returns = benchmark_prices.pct_change()

    ff_factors = process_fama_french_factors(ff_url)
    rf_rate = ff_factors['RF'].iloc[-1] if 'RF' in ff_factors.columns else None

    return prices, returns, benchmark_prices, benchmark_returns, rf_rate, ff_factors
