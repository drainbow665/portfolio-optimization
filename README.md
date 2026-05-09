 Multi-Asset Portfolio Optimization via Monte Carlo Simulation
---

Problem Statement
Investors face a trade-off between risk and return when allocating capital across multiple assets. The objective of this project is to determine the optimal portfolio allocation that maximizes risk-adjusted return using the Sharpe Ratio.

This project answers:
- What combination of assets gives the highest Sharpe Ratio?
- How does optimization compare with a simple equal-weight portfolio?
- How does diversification affect portfolio performance?

---
 Dataset Description
The dataset is obtained from Yahoo Finance using the yfinance Python library.

Assets used:
- SPY → S&P 500 ETF (Equity)
- QQQ → NASDAQ-100 ETF (Technology)
- VTI → Total Market ETF
- GLD → Gold ETF (Commodity)
- AGG → Bond ETF (Fixed Income)

Time period:
- 5 years of daily data

Features computed:
- Daily returns
- Mean returns (annualized)
- Covariance matrix
- Correlation matrix
- Volatility

 Methodology
The project is based on Markowitz Mean-Variance Optimization.

Portfolio Return:
E(Rp) = wᵀμ

Portfolio Risk:
σ = √(wᵀΣw)

Sharpe Ratio:
SR = (Rp − Rf) / σ

Monte Carlo Simulation:
- 10,000 random portfolios generated
- Random weights assigned to assets
- Each portfolio evaluated for return, risk, and Sharpe Ratio

Steps:
1. Download historical price data
2. Calculate returns and covariance matrix
3. Generate random portfolio weights
4. Compute portfolio performance
5. Identify optimal portfolios:
   - Maximum Sharpe Ratio
   - Minimum Volatility
  

The code is structured into the following parts:

1. Import libraries (NumPy, Pandas, Matplotlib, yfinance)
2. Configuration (assets, simulation size, dates)
3. Data download from Yahoo Finance
4. Return and risk calculations
5. Monte Carlo simulation (10,000 portfolios)
6. Identification of optimal portfolios
7. Efficient Frontier construction
8. Visualization (7 graphs)
9. Summary comparison

 Results and Insights

- The Maximum Sharpe Portfolio outperforms the equal-weight portfolio significantly.
- Sharpe Ratio improves by approximately 40–80%.
- GLD (Gold) has low correlation with equities, improving diversification.
- SPY, QQQ, and VTI are highly correlated, reducing diversification benefits.
- AGG provides stability but lower returns.

Conclusion:
Portfolio optimization improves risk-adjusted returns by reducing volatility through diversification.
