import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def download_data(ticker, start_date, end_date, interval='1d'):
    """Downloads historical price data at a specified interval (e.g., daily)."""
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check the ticker or date range.")
    
    # Use 'Adj Close' if available, otherwise fallback to 'Close'
    column_name = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    if column_name not in data.columns:
        raise KeyError(f"Neither 'Adj Close' nor 'Close' column found for ticker {ticker}. Available columns: {data.columns}")

    adj_close = data[column_name]
    adj_close.index = pd.to_datetime(adj_close.index)  # Ensure datetime index
    adj_close.name = ticker  # Rename the series to match the ticker
    return adj_close

def calculate_returns(prices):
    """Calculates returns from price data."""
    return prices.pct_change().dropna()

def calculate_beta_regression(stock_returns, market_returns):
    """Calculates beta using linear regression."""
    X = market_returns.values.reshape(-1, 1)  # Independent variable
    y = stock_returns.values  # Dependent variable
    model = LinearRegression().fit(X, y)
    beta = model.coef_.item()  # Use .item() to extract the scalar value
    return beta

def calculate_beta_covariance(stock_returns, market_returns):
    """Calculates beta using covariance and variance."""
    # Ensure both inputs are pandas Series
    if not isinstance(stock_returns, pd.Series):
        stock_returns = stock_returns.squeeze()  # Convert to Series
    if not isinstance(market_returns, pd.Series):
        market_returns = market_returns.squeeze()  # Convert to Series

    # Ensure the lengths match
    aligned_returns = pd.concat([stock_returns, market_returns], axis=1).dropna()
    stock_returns = aligned_returns.iloc[:, 0]  # Reassign aligned series
    market_returns = aligned_returns.iloc[:, 1]

    # Calculate covariance and variance
    covariance = stock_returns.cov(market_returns)  # Covariance of stock and market returns
    market_variance = market_returns.var()  # Variance of market returns
    beta_cov = covariance / market_variance
    return beta_cov

def plot_stock_vs_market_returns(stock_returns, market_returns, stock_ticker, market_ticker):
    """Generates a scatter plot of stock vs. market returns with regression line."""
    # Perform linear regression
    X = market_returns.values.reshape(-1, 1)
    y = stock_returns.values
    model = LinearRegression().fit(X, y)
    slope = model.coef_.item()  # Extract scalar from array
    intercept = model.intercept_.item()  # Extract scalar from array
    regression_line = slope * market_returns + intercept

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(market_returns, stock_returns, alpha=0.5, label="Daily Returns")
    plt.plot(market_returns, regression_line, color="orange", linewidth=2, label=f"y = {slope:.4f}x + {intercept:.4f}")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # Add labels, title, and legend
    plt.title(f"Stock Returns vs. Market Returns\n({stock_ticker} vs. {market_ticker})", fontsize=14)
    plt.xlabel("Market Returns", fontsize=12)
    plt.ylabel("Stock Returns", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()

def main():
    # Parameters
    stock_ticker = "AVGO"
    market_ticker = "^GSPC"
    risk_free_ticker = "^TNX"  # 10-Year Treasury rate
    start_date = "2018-01-01" # "2014-11-24"
    end_date = "2024-12-31"

    tickers = {
        "Stock Prices": stock_ticker,
        "Market Prices": market_ticker,
        "Risk-Free Data": risk_free_ticker
    }

    print("Downloading data...")
    data_dict = {}
    with tqdm(total=len(tickers), desc="Progress", ncols=80) as pbar:
        for name, ticker in tickers.items():
            try:
                data_dict[name] = download_data(ticker, start_date, end_date, interval='1d')
                print(f"Downloaded {ticker} successfully.")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
            pbar.update(1)

    # Extract downloaded data
    stock_prices = data_dict.get("Stock Prices")
    market_prices = data_dict.get("Market Prices")
    risk_free_data = data_dict.get("Risk-Free Data")
    risk_free_data = risk_free_data.dropna()
    
    # Ensure risk-free data is a Series
    risk_free_data = risk_free_data.iloc[:, 0]  # Take first column of DataFrame

    # Calculate daily returns
    stock_returns = calculate_returns(stock_prices)
    market_returns = calculate_returns(market_prices)

    # Calculate arithmetic mean for stock and market returns
    arithmetic_mean_stock = stock_returns.mean()
    arithmetic_mean_market = market_returns.mean()

    # Ensure scalars
    arithmetic_mean_stock = arithmetic_mean_stock.iloc[0] if isinstance(arithmetic_mean_stock, pd.Series) else float(arithmetic_mean_stock)
    arithmetic_mean_market = arithmetic_mean_market.iloc[0] if isinstance(arithmetic_mean_market, pd.Series) else float(arithmetic_mean_market)

    # Annualize the arithmetic mean
    annualized_stock_return = arithmetic_mean_stock * 252  # 252 trading days
    annualized_market_return = arithmetic_mean_market * 252   # 252 trading days 

    # Print the annualized arithmetic returns
    print(f"Annualized Stock Return (Arithmetic) ({stock_ticker}): {annualized_stock_return * 100:.2f}%")
    print(f"Annualized Market Return (Arithmetic) ({market_ticker}): {annualized_market_return * 100:.2f}%")
    
    # Calculate beta using linear regression
    beta_regression = calculate_beta_regression(stock_returns, market_returns)
    print(f"Beta of {stock_ticker} relative to {market_ticker} (linear regression): {beta_regression:.4f}")

    # Calculate beta using covariance/variance
    beta_covariance = calculate_beta_covariance(stock_returns, market_returns)
    print(f"Beta of {stock_ticker} relative to {market_ticker} (covariance/variance): {beta_covariance:.4f}")

    # Calculate the standard deviation of stock returns
    stock_std_dev = stock_returns.std()  # Standard deviation of daily returns
    stock_std_dev = stock_std_dev.iloc[0] if isinstance(stock_std_dev, pd.Series) else float(stock_std_dev)
    annualized_std_dev = stock_std_dev * np.sqrt(252)  # Annualize the standard deviation

    # Print the result
    print(f"Annualized Standard Deviation of {stock_ticker} Equity: {annualized_std_dev * 100:.2f}%")

    # Call the plotting function
    #plot_stock_vs_market_returns(stock_returns, market_returns, stock_ticker, market_ticker)
    
    # Convert most recent 10Y Treasury yield to decimal
    latest_rf_rate = risk_free_data.iloc[-1] / 100

    # CAPM cost of equity
    cost_of_equity = latest_rf_rate + beta_regression * (annualized_market_return - latest_rf_rate)

    print(f"Risk-Free Rate: {latest_rf_rate * 100:.2f}%")
    print(f"Estimated Cost of Equity (CAPM): {cost_of_equity * 100:.2f}%")

if __name__ == "__main__":
    main()
