# Libraries

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Pick an ETF
ticker = "SPY"

class BlackScholes:

    """
    BS options pricing engine for European calls and puts

    Parameters (All floats):
        S: Current Spot Price
        K: Strike Price
        r: Annualized Risk Free Rate
        T: Time to Expiry in Years 
        sigma: Annualized Volatility

    """

    def __init__(self, S: float, K: float, r: float, T: float, sigma:float):
        self.S = S
        self.K = K
        self.r = r
        self.T = T 
        self.sigma = sigma
        self.d1 = self._d1()
        self.d2 = self._d2()

    # Intermediate functions D1 and D2
    def _d1(self) -> float:
        y =  (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2)
               * self.T) / (self.sigma * np.sqrt(self.T))

        return y

    def _d2(self) -> float:
        return self._d1() - self.sigma * np.sqrt(self.T)

    # -----------------------------------------------Pricing -------------------------------------------

    def call_price(self) -> float:
     return (self.S * norm.cdf(self.d1) - 
              self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self) -> float:
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * norm.cdf(-self.d1))

    # -----------------------------------------------Greeks----------------------------------------------

    def delta(self, option_type: str = "call") -> float:
        """Rate of change of option price with respect to spot price"""
        if option_type == "call":
            return norm.cdf(self.d1)
        return norm.cdf(self.d1) - 1

    def gamma(self) -> float:
        """Rate of change of Delta with respect to spot price for calls and puts"""
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Sensitivity to volatility. Returns value per 1% change in volatility"""
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T) * 0.01

    def theta(self, option_type: str = "call") -> float:
        """Time decay per day"""
        common = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == "call":
            return (common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 365
        return (common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 365

    def rho(self, option_type: str = "call") -> float:
        """Sensitivity to risk-free rate. Returns value per 1% change in rate"""
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) * 0.01
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) * 0.01

    def summary(self, option_type: str = "call") -> None:
        """Prints a summary of price and all Greeks."""
        price = self.call_price() if option_type == "call" else self.put_price()
        print(f"\n{'='*40}")
        print(f"Black-Scholes {option_type.upper()} Option Summary")
        print(f"{'='*40}")
        print(f"Spot:      {self.S:.2f}  |  Strike: {self.K:.2f}")
        print(f"Rate:      {self.r*100:.1f}%  |  Vol:    {self.sigma*100:.1f}%")
        print(f"Expiry:    {self.T:.2f} years")
        print(f"{'─'*40}")
        print(f"Price:     {price:.4f}")
        print(f"Delta:     {self.delta(option_type):.4f}")
        print(f"Gamma:     {self.gamma():.4f}")
        print(f"Vega:      {self.vega():.4f}  (per 1% vol)")
        print(f"Theta:     {self.theta(option_type):.4f}  (per day)")
        print(f"Rho:       {self.rho(option_type):.4f}  (per 1% rate)")
        print(f"{'='*40}\n")

# --------------------------------------------- MONTE CARLO VALIDATION-------------------------------------------

def monte_carlo_price(S, K, r, T, sigma, option_type="call", n_sims=100_000) -> float:
    """
    Price a European option through Monte Carlo simulation using GBM.
    Useful for validating Black-Scholes analytically
    """
    np.random.seed(42)
    Z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    return np.exp(-r * T) * np.mean(payoffs)


def mc_convergence(S, K, r, T, sigma, option_type="call", max_sims=100_000):
    """Plot Monte Carlo convergence toward Black-Scholes price"""
    bs = BlackScholes(S, K, r, T, sigma)
    bs_price = bs.call_price() if option_type == "call" else bs.put_price()
    
    sim_counts = np.logspace(2, 5, 50).astype(int)
    mc_prices = [monte_carlo_price(S, K, r, T, sigma, option_type, n) for n in sim_counts]
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(sim_counts, mc_prices, label="Monte Carlo", color="steelblue")
    plt.axhline(bs_price, color="crimson", linestyle="--", label=f"Black-Scholes: {bs_price:.4f}")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.title(f"Monte Carlo Convergence — European {option_type.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150)
    plt.show()

# ----------------------------------------------Real Market Data --------------------------------------



def get_real_data(ticker):
    stock = yf.Ticker(ticker)

    # Some ETFs (like S&P) don't have currentPrice in info, use last close instead
    try:
        S = stock.info["currentPrice"]
    except KeyError:
        try:
            S = stock.info["regularMarketPrice"]
        except KeyError:
            S = stock.history(period="1d")["Close"].iloc[-1]
    
    # Historical volatility / Annualized std of log returns
    hist = stock.history(period="1y")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)  # annualize
    
    # Options chain for real market prices
    expiries = stock.options  # available expiry dates
    chain = stock.option_chain(expiries[2])  # pick an expiry
    calls = chain.calls
    puts = chain.puts

    calls = calls[
        (calls["strike"] > S * 0.915) &      # within 10% below spot
        (calls["strike"] < S * 1.10) &      # within 10% above spot
        (calls["lastPrice"] > 0.05) &       # exclude near-zero prices
        (calls["impliedVolatility"] > 0.01) # exclude illiquid options
    ]
    puts = puts[
        (puts["strike"] > S * 0.915) &
        (puts["strike"] < S * 1.10) &
        (puts["lastPrice"] > 0.05) &
        (puts["impliedVolatility"] > 0.01)
    ]

    return S, sigma, calls, puts

S, sigma, calls, puts = get_real_data("SPY")
print(f"Spot: {S}, Historical Vol: {sigma:.2%}")
print(calls[["strike", "lastPrice", "impliedVolatility"]].head(10))

def compare_iv(ticker, r=0.05):
    stock = yf.Ticker(ticker)

    # Same as in get_real_data, might not have current price
    try:
            S = stock.info["currentPrice"]
    except KeyError:
            try:
                S = stock.info["regularMarketPrice"]
            except KeyError:
                S = stock.history(period="1d")["Close"].iloc[-1]
    
    # Historical vol as the sigma estimate
    hist = stock.history(period="1y")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hist_vol = log_returns.std() * np.sqrt(252)
    
    # Get nearest expiry options
    exp = stock.options[1]
    T = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365, 1/365)
    calls = stock.option_chain(exp).calls

    # Filters 
    calls = calls[
        (calls["strike"] > S * 0.915) &      # within 10% below spot
        (calls["strike"] < S * 1.10) &      # within 10% above spot
        (calls["lastPrice"] > 0.05) &       # exclude near-zero prices
        (calls["impliedVolatility"] > 0.01) # exclude illiquid options
        ]
    
    results = []
    for _, row in calls.iterrows():
        K = row["strike"]
        market_price = row["lastPrice"]
        market_iv = row["impliedVolatility"]
        
        # BS price using historical volatility
        bs = BlackScholes(S, K, r, T, hist_vol)
        bs_price = bs.call_price()
        
        # Implied vol from market price
        iv = implied_volatility(market_price, S, K, r, T)
        
        results.append({
            "Strike": K,
            "Market Price": market_price,
            "BS Price (hist vol)": round(bs_price, 2),
            "Market IV": round(market_iv, 4),
            "IV": round(iv, 4) if not np.isnan(iv) else None
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

def plot_iv_surface(ticker, r=0.05):
    stock = yf.Ticker(ticker)
    # Same as in get_real_data, might not have current price
    try:
                S = stock.info["currentPrice"]
    except KeyError:
            try:
                S = stock.info["regularMarketPrice"]
            except KeyError:
                S = stock.history(period="1d")["Close"].iloc[-1]
    expiries = stock.options[:6]  # first 6 expiries
    
    strikes_all, expiries_all, ivs_all = [], [], []
    
    for exp in expiries:
        chain = stock.option_chain(exp)
        calls = chain.calls
        calls = calls[(calls["strike"] > S * 0.8) & 
                     (calls["strike"] < S * 1.2) &
                     (calls["impliedVolatility"] > 0)]
        
        # Time to expiry in years
        from datetime import datetime
        T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365
        
        for _, row in calls.iterrows():
            strikes_all.append(row["strike"])
            expiries_all.append(T)
            ivs_all.append(row["impliedVolatility"])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(strikes_all, expiries_all, ivs_all, c=ivs_all, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to Expiry (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker} Implied Volatility Surface")
    plt.tight_layout()
    plt.savefig("iv_surface.png", dpi=150)
    plt.show()

#-----------------------------------------------Implied Volatility -------------------------------------

def implied_volatility(market_price, S, K, r, T, option_type="call") -> float:
    """
    Back out implied volatility from a market price using Brent's method.
    Returns implied volatility as a decimal
    """
    def objective(sigma):
        bs = BlackScholes(S, K, r, T, sigma)
        model_price = bs.call_price() if option_type == "call" else bs.put_price()
        return model_price - market_price
    
    try:
        iv = brentq(objective, 1e-4, 5.0)
        return iv
    except ValueError:
        try:
            iv = brentq(objective, 1e-6, 20.0)
            return iv
        except ValueError:
            return np.nan

# ------------------------------------------------Visualizations-----------------------------------------

def plot_price_and_greeks(K, r, T, sigma, option_type="call"):
    """Plot option price, Delta, Gamma, Theta, and Vega across spot prices"""
    spot_range = np.linspace(K * 0.5, K * 1.5, 200)
    
    prices, deltas, gammas, thetas, vegas = [], [], [], [], []
    
    for S in spot_range:
        bs = BlackScholes(S, K, r, T, sigma)
        prices.append(bs.call_price() if option_type == "call" else bs.put_price())
        deltas.append(bs.delta(option_type))
        gammas.append(bs.gamma())
        thetas.append(bs.theta(option_type))
        vegas.append(bs.vega())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Black-Scholes {option_type.capitalize()} — Greeks vs Spot Price", fontsize=14)
    
    plots = [
        (prices,  "Option Price",  "steelblue"),
        (deltas,  "Delta",         "darkorange"),
        (gammas,  "Gamma",         "green"),
        (thetas,  "Theta (daily)", "crimson"),
        (vegas,   "Vega (per 1%)", "purple"),
    ]
    
    for ax, (data, label, color) in zip(axes.flat, plots):
        ax.plot(spot_range, data, color=color, linewidth=2)
        ax.axvline(K, color="black", linestyle="--", alpha=0.4, label="Strike")
        ax.set_xlabel("Spot Price")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    
    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig("greeks.png", dpi=150)
    plt.show()


def plot_theta_decay(S, K, r, sigma, option_type="call"):
    """Show how option price decays as expiry approaches"""
    time_range = np.linspace(0.01, 1.0, 200)
    prices = [BlackScholes(S, K, r, T, sigma).call_price() 
              if option_type == "call" 
              else BlackScholes(S, K, r, T, sigma).put_price() 
              for T in time_range]
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_range, prices, color="crimson", linewidth=2)
    plt.xlabel("Time to Expiry (Years)")
    plt.ylabel("Option Price")
    plt.title(f"Theta Decay — European {option_type.capitalize()}")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("theta_decay.png", dpi=150)
    plt.show()

# -------------------------------------------- Main ------------------------------------------

if __name__ == "__main__":
    
    # Example: ATM call option
    S, K, r, T, sigma = 100, 100, 0.05, 0.5, 0.20
    
    bs = BlackScholes(S, K, r, T, sigma)
    bs.summary("call")
    bs.summary("put")
    
    # Monte Carlo validation
    mc_call = monte_carlo_price(S, K, r, T, sigma, "call", 100_000)
    print(f"BS Call:  {bs.call_price():.4f}")
    print(f"MC Call:  {mc_call:.4f}")
    print(f"Diff:     {abs(bs.call_price() - mc_call):.6f}\n")
    
    # Implied volatility (Verifier Data)
    market_price = 10.50
    iv = implied_volatility(market_price, S, K, r, T, "call")
    print(f"Market Price: {market_price}")
    print(f"Implied Vol:  {iv*100:.2f}%\n")

    # Real Market Data 

    print(f"\nFetching real market data for {ticker}...")
    
    S_real, sigma_real, calls, puts = get_real_data(ticker)
    print(f"Spot:           ${S_real:.2f}")
    print(f"Historical Vol: {sigma_real:.2%}\n")
    
    # Compare BS model prices vs real market prices
    print("Comparing Black-Scholes vs Market Prices:")
    df = compare_iv(ticker, r=0.05)


    # Volume Risk Premium
    print(f"\nVolatility Risk Premium: {df['IV'].mean() - df['Market IV'].mean():.2%}")
    print("Positive premium means market is pricing more future uncertainty than historical volatility  implies")
    
    # Plots
    plot_price_and_greeks(K, r, T, sigma, "call")
    plot_theta_decay(S, K, r, sigma, "call")
    mc_convergence(S, K, r, T, sigma, "call")
    plot_iv_surface(ticker, r=0.05)
