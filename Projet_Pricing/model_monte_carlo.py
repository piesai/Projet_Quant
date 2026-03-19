import numpy as np
from scipy.stats import norm

# Parameters
S0 = 100       # Initial stock price
K = 150        # Strike price
T = 1.0        # Time to maturity (years)
r = 0.05       # Risk-free rate
sigma = 0.2    # Volatility
n_paths = 100000  # Number of simulated paths
seed = 42      # Random seed for reproducibility

# Black-Scholes formula for European call
def black_scholes_call(S0, K, T, r, sigma):
    """
    Compute the price of a European call option using the Black-Scholes formula.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Monte Carlo simulation for European call
def monte_carlo_call(S0, K, T, r, sigma, n_paths):
    """
    Price a European call option using Monte Carlo simulation.
    """
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)  # Generate random normal variables
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # Simulate terminal stock prices
    payoff = np.maximum(ST - K, 0)  # Call option payoff
    call_price = np.exp(-r * T) * np.mean(payoff)  # Discounted average payoff
    return call_price

# Compute prices
mc_price = monte_carlo_call(S0, K, T, r, sigma, n_paths)
bs_price = black_scholes_call(S0, K, T, r, sigma)

# Print results
print(f"Monte Carlo European Call Price: {mc_price:.4f}")
print(f"Black-Scholes European Call Price: {bs_price:.4f}")