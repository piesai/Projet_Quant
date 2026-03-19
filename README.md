# Projet_Quant
Testing things about quantitative finance.

---

## Projet_Pricing
After reading John C. Hull's *"Options, Futures, and Other Derivatives"*, I wanted to test if theory aligns with practice using real-world financial data. In this project, I explore different pricing methods for options, including Black-Scholes-Merton (BSM), Monte Carlo simulations, and Binomial Trees.

---

### Naive Models

#### Model_BSM
- **Objective**: Implement the Black-Scholes-Merton (BSM) model for pricing calls on **AAPL** stock.
- **Implementation**: Chosen for its simplicity and foundational role in options pricing.
- **Results**:
  - **In-the-Money Options**: The model performs well, as evidenced by the graphs.
  - **Out-of-the-Money Options**: Struggles due to:
    - BSM being designed for **European options**, while tested on **American options**.
    - The assumption of **constant volatility**, which is unrealistic in practice.

#### Model_Binomial_Tree
- **Objective**: Extend the BSM model to account for dividends and American option features.
- **Results**:
  - **Very In-the-Money (K << S)**: Accurate pricing with a high volatility skew, reflecting real market behavior.
  - **Very Out-of-the-Money (K >> S)**: Underpriced compared to real market data. This discrepancy arises because:
    - Traders price in a risk premium for these options.
    - Without this premium, the model predicts near-zero prices, which is unrealistic.

#### Update: Using "Ask" Price
- **Change**: Switched from "lastPrice" to "ask" for option pricing.
- **Improvement**: Results improved significantly for **implied volatility**.
- **Observations**:
  - **BSM**: Better for in-the-money options but still struggles out-of-the-money.
  - **Binomial Tree**: Consistent error across strikes, making it more reliable for out-of-the-money options.
- **Conclusion**: These models work well for implied volatility but fail to predict option prices without it. A **volatility model** is needed to capture the volatility skew for long-term strategies.

---

## Projet_PM
Exploring portfolio management strategies.


#### Projet_Markowitz
- **Objective**: Implement Harry Markowitz's "Optimal Portfolio" theory to construct efficient portfolios.

---

## Project_Pricing_Heston
- **Objective**: Calibrate the Heston Model to real world data.
