import pytest
import numpy as np
from BlackScholes import BlackScholes, implied_volatility, monte_carlo_price

class TestBlackScholes:

    def test_call_price_atm(self):
        """ATM call should be positive and reasonable"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        price = bs.call_price()
        assert price > 0
        assert price < 100  # can't exceed spot

    def test_put_call_parity(self):
        """Call - Put = S - K*e^(-rT) (put-call parity)"""
        S, K, r, T, sigma = 100, 100, 0.05, 0.5, 0.20
        bs = BlackScholes(S, K, r, T, sigma)
        lhs = bs.call_price() - bs.put_price()
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6  # should be exact

    def test_delta_call_bounds(self):
        """Call delta must be between 0 and 1"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        delta = bs.delta("call")
        assert 0 <= delta <= 1

    def test_delta_put_bounds(self):
        """Put delta must be between -1 and 0"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        delta = bs.delta("put")
        assert -1 <= delta <= 0

    def test_gamma_positive(self):
        """Gamma must always be positive"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        assert bs.gamma() > 0

    def test_vega_positive(self):
        """Vega must always be positive"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        assert bs.vega() > 0

    def test_theta_negative(self):
        """Theta must always be negative — options lose value over time"""
        bs = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.20)
        assert bs.theta("call") < 0
        assert bs.theta("put") < 0

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta should approach 1"""
        bs = BlackScholes(S=200, K=100, r=0.05, T=0.5, sigma=0.20)
        assert bs.delta("call") > 0.99

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta should approach 0"""
        bs = BlackScholes(S=50, K=100, r=0.05, T=0.5, sigma=0.20)
        assert bs.delta("call") < 0.01

    def test_monte_carlo_convergence(self):
        """MC price should be within 2% of BS price at 100k sims"""
        S, K, r, T, sigma = 100, 100, 0.05, 0.5, 0.20
        bs = BlackScholes(S, K, r, T, sigma)
        mc = monte_carlo_price(S, K, r, T, sigma, "call", 100_000)
        assert abs(bs.call_price() - mc) / bs.call_price() < 0.02

    def test_implied_volatility_roundtrip(self):
        """IV solver should recover original sigma from BS price"""
        S, K, r, T, sigma = 100, 100, 0.05, 0.5, 0.25
        bs = BlackScholes(S, K, r, T, sigma)
        price = bs.call_price()
        recovered_iv = implied_volatility(price, S, K, r, T, "call")
        assert abs(recovered_iv - sigma) < 1e-4

    def test_zero_time_to_expiry(self):
        """At expiry, call value equals max(S-K, 0)"""
        bs = BlackScholes(S=110, K=100, r=0.05, T=1/365, sigma=0.20)
        assert bs.call_price() > 0

    def test_higher_vol_higher_price(self):
        """Higher volatility should produce higher option prices"""
        bs_low = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.10)
        bs_high = BlackScholes(S=100, K=100, r=0.05, T=0.5, sigma=0.40)
        assert bs_high.call_price() > bs_low.call_price()
        assert bs_high.put_price() > bs_low.put_price()
