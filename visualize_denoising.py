import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import fetch_yfinance_data
from cov_denoising import getCovMatrix, getPCA, mpPDF, fitKDE, findMaxEval, cov2corr, denoisedCorr

# fetch Stock Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'KR', 'BRK-B', 'META', 'SPOT', 'NFLX', 'PYPL', 'IBM', 'CSCO', 'GILD', 'ABBV', 'ABT', 'LNG', 'JNJ', 'AVGO', 'MDT', 'LLY', 'CHD', 'BAC', 'KNX', 'ADP', 'AFL', 'CNI', 'CVX', 'HD']  # Portfolio tickers
start_date = '2000-01-01'
end_date = '2024-01-01'

prices, returns = fetch_yfinance_data(tickers, start_date, end_date)

print(prices)

returns = returns.dropna()

print(returns)

# plot adjusted stock prices over time
plt.figure(figsize=(12, 6))
for stock in prices.columns:
    plt.plot(prices.index, prices[stock], label=stock)

plt.xlabel("Date")
plt.ylabel("Adjusted Price")
plt.title("Stock Prices Over Time")
plt.legend()
plt.grid()
plt.show()


# compute covariance and correlation matrices
cov_matrix = getCovMatrix(returns)
corr_matrix = cov2corr(cov_matrix)


print(f"Shape of correlation matrix: {corr_matrix.shape}")
print(f"Is symmetric? {np.allclose(corr_matrix, corr_matrix.T, atol=1e-8)}")
print(f"Any NaN values? {np.isnan(corr_matrix).any()}")
print(f"Any Inf values? {np.isinf(corr_matrix).any()}")
print(f"Minimum Eigenvalue (before PCA): {np.linalg.eigvalsh(corr_matrix).min()}")

# perform PCA to extract eigenvalues
eVal, eVec = getPCA(corr_matrix)  

print("Eigenvalues before denoising:", np.diag(eVal))
print(f"Min Eigenvalue: {np.min(np.diag(eVal))}")
print(f"Mean Eigenvalue: {np.mean(np.diag(eVal))}")
print(f"Max Eigenvalue: {np.max(np.diag(eVal))}")
print(f"Eigenvalue Shape: {eVal.shape}")

# fit the Marčenko-Pastur distribution
T, N = returns.shape  # Time periods and number of assets
q = T / N  # Ratio of observations to variables

eMax, var = findMaxEval(np.diag(eVal), q, bWidth=0.01)

mp_pdf = mpPDF(var, q, pts=1000)

nFacts = np.sum(eVal > eMax) # signal eigenvalues are those above MP threshold
nFacts = max(nFacts, int(N * 0.2))  # Ensure at least 20% of eigenvalues are treated as signal

print(f"nFacts (signal eigenvalues): {nFacts}")
print(f"Total number of eigenvalues: {N}")


plt.figure(figsize=(8, 5))
plt.hist(np.diag(eVal), bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(np.diag(eVal)), color='red', linestyle="dashed", label="Mean Eigenvalue")
plt.title("Eigenvalue Distribution Before Denoising")
plt.xlabel("Eigenvalue ($\lambda$)")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

empirical_pdf = fitKDE(np.diag(eVal), bWidth=0.05, x=mp_pdf.index.values)

plt.figure(figsize=(8, 5))

# plot histogram
sns.histplot(np.diag(eVal), bins=30, stat="density", kde=False, edgecolor='black', alpha=0.6, label="Empirical Eigenvalues")

# overlay the theoretical Marčenko-Pastur PDF
plt.plot(mp_pdf.index, mp_pdf, linestyle='--', color="blue", label="Theoretical MP PDF")

# overlay Empirical KDE
plt.plot(empirical_pdf.index, empirical_pdf, color='red', label="Empirical KDE (Red)")

# mark the MP threshold
plt.axvline(eMax, color='black', linestyle='dotted', label=f"MP Threshold ($\lambda_{max}$ = {eMax:.2f})")

plt.xlabel("Eigenvalue ($\lambda$)")
plt.ylabel("Density")
plt.title("Comparing Histogram, KDE, and MP PDF")
plt.legend()
plt.grid()
plt.show()

# denoised correlation matrix
corr1 = denoisedCorr(eVal, eVec, nFacts) 

eVal1, eVec1 = getPCA(corr1)

# plot comparison of eigenvalues before and after denoising
plt.figure(figsize=(10, 6))

# plot original eigenvalues
plt.plot(range(1, len(eVal) + 1), np.sort(np.diag(eVal))[::-1], label="Original Eigenvalues", color="blue")

# plot denoised eigenvalues
plt.plot(range(1, len(eVal1) + 1), np.sort(np.diag(eVal1))[::-1], linestyle="dashed", color="orange", label="Denoised Eigenvalues")

# set log scale for y-axis
plt.yscale("log")

# labels and title
plt.xlabel("Eigenvalue Number")
plt.ylabel("Eigenvalue (log-scale)")
plt.title("Comparison of Eigenvalues Before and After Denoising")

# legend and grid
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# show plot
plt.show()
