# Black-Scholes Options Pricing Engine

A full Black-Scholes derivatives pricing toolkit built in Python. Prices European calls and puts 
analytically, computes all five Greeks, validates prices via Monte Carlo simulation, back-solves 
for implied volatility using Brent's numerical method, and connects to live market data to 
construct the volatility smile and measure the variance risk premium.

## Motivation

This project was built to develop intuition for derivatives pricing and risk measurement. 
The Greeks are central to how traders hedge options positions in practice. Delta hedging, Gamma 
scalping, and Vega management all depend on accurate and fast Greeks computation. The implied 
volatility solver reverses the model to extract market-implied expectations from observed prices, 
which is how practitioners actually use Black-Scholes day to day.

Monte Carlo simulation serves as an independent validation of the closed-form Black-Scholes 
solution. Because both methods price the same derivative under identical assumptions (GBM, constant 
vol, no dividends), convergence of the Monte Carlo price toward the analytical price as simulations 
increase confirms the correctness of the implementation. The comparison also illustrates why 
closed-form solutions are preferred in practice. Speed matters on a trading desk. For exotic 
derivatives where no closed form exists (barriers, Asians, lookbacks), Monte Carlo becomes the 
primary pricing method.

## Real Market Data

The engine connects to live market data via yfinance, pulling real spot prices, historical 
volatility estimates from annualized log returns (252 trading days), and actual options chains 
with market prices across strikes and expiries. This allows direct comparison of model prices 
against market quotes and construction of the implied volatility surface, revealing the 
volatility smile that Black-Scholes cannot explain, and where more advanced models like Heston 
or SABR become necessary.

## Key Concepts

**d1 and d2** are the core intermediate values in the Black-Scholes formula. d1 measures how 
far the spot price is from the strike, adjusted for expected drift and volatility over the 
option's life. N(d1) gives the option's Delta or the probability that the option ends in the 
money under the stock measure. d2 adjusts d1 by one standard deviation of the stock's return 
(σ√T) and represents the risk-neutral probability that the option expires in the money. N(d2) 
is the probability used to discount the expected strike payment.

The call price formula — S·N(d1) − K·e^(−rT)·N(d2) — can be read as: the expected value of 
receiving the stock if the option expires in the money, minus the discounted expected cost of 
paying the strike.

**Greeks** measure how sensitive the option price is to changes in inputs:
- **Delta** — sensitivity to spot price movement
- **Gamma** — rate of change of Delta (convexity)
- **Vega** — sensitivity to volatility (per 1% change)
- **Theta** — time decay per calendar day
- **Rho** — sensitivity to the risk-free rate (per 1% change)

**Implied Volatility** inverts the model given an observed market price, Brent's method 
numerically solves for the volatility that makes the Black-Scholes price match the market. 
This is how the volatility surface is constructed in practice.

**Monte Carlo** prices the option by simulating thousands of stock price paths under Geometric 
Brownian Motion and averaging the discounted payoffs. It converges to the Black-Scholes price 
as the number of simulations increases, validating the closed-form solution.

**European vs American Options** this engine prices European-style options, which can only 
be exercised at expiry. American options can be exercised at any point before expiry and require 
numerical methods such as binomial trees due to the early exercise premium. SPY options are 
technically American-style, introducing a small early exercise premium not captured here — a 
natural extension of this project.

## Key Findings

**Volatility Smile**
Market implied volatility is highest for deep in-the-money options and decreases toward 
at-the-money strikes, revealing the well-known volatility smile. Black-Scholes assumes constant 
volatility across all strikes, which is clearly violated in practice. This systematic mispricing 
motivates more advanced models like Heston (stochastic volatility) and SABR.

**Variance Risk Premium**
Using live SPY options data, the model identifies a variance risk premium of approximately 7.75% 
— the spread between market-implied volatility and historically realized volatility. This premium 
exists because option sellers demand compensation for bearing volatility risk, and investors 
willingly pay above fair value for downside protection. The variance risk premium is one of the 
most well-documented phenomena in derivatives markets and forms the basis for volatility 
arbitrage strategies.

**Near-ATM Accuracy**
Black-Scholes prices closely match market prices for near-the-money options (within 1-2% of 
spot), confirming the model's practical utility in this range despite its theoretical limitations.

**Model Limitations**
The assumption of constant volatility is clearly violated by the observed smile. The log-normal 
return assumption understates tail risk relative to what markets actually price in. Historical 
volatility (12.88% for SPY) significantly understates market-implied volatility, consistent with 
the well-documented variance risk premium. SPY options also carry a small early exercise premium 
not captured by the European formula.

## Visualizations

- `greeks.png` — option price and all five Greeks across spot prices
- `theta_decay.png` — option price decay as expiry approaches (read right to left)
- `mc_convergence.png` — Monte Carlo convergence toward Black-Scholes price
- `iv_surface.png` — implied volatility surface across strikes and expiries

## Installation

```bash
pip install numpy scipy matplotlib pandas yfinance
```

## Usage

```python
from black_scholes import BlackScholes, implied_volatility, monte_carlo_price

# Price an ATM call option
bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
bs.summary("call")

# Monte Carlo validation
mc_price = monte_carlo_price(100, 100, 0.05, 0.5, 0.20, "call", 100_000)

# Implied volatility from market price
iv = implied_volatility(market_price=10.50, S=100, K=100, r=0.05, T=0.5)
print(f"Implied Vol: {iv*100:.2f}%")

# Live market data
S, sigma, calls, puts = get_real_data("SPY")
df = compare_iv("SPY", r=0.05)
```

## Project Structure

- black_scholes.py # Core pricing engine, Greeks, Monte Carlo, IV solver, market data
- greeks.png # Option price and Greeks across spot prices
- theta_decay.png # Option price decay as expiry approaches
- mc_convergence.png # Monte Carlo convergence to Black-Scholes price
- iv_surface.png # Implied volatility surface across strikes and expiries

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- yfinance

## Future Extensions

- Binomial tree pricer for American options and early exercise premium
- Heston stochastic volatility model to capture the volatility smile
- SABR model for interest rate derivatives
- Real-time options chain streaming
- Portfolio-level Greeks aggregation and Delta hedging simulation

## Author

Theodore Van Cappellen
BSc Economics with Minor in Statistics, University of Toronto
CFA Level I Candidate
