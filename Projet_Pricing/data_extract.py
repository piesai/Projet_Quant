import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#on regarde l'évolution du cours de l'action
ticker_symbol = 'AAPL'  # Example: Apple Inc.
ticker = yf.Ticker(ticker_symbol)
cours_action = ticker.history(period = '5y')
cours_fermeture = cours_action["Close"]
print(cours_fermeture.keys()[1200])
#plt.plot([k for k in range(1256)],cours_fermeture)
plt.xlabel("Jours sur les 5 dernières années depuis 19 Juillet 2025")
plt.ylabel("cours de l'action")
#plt.show()

#on visualise les options sur cette action 
options = ticker.options
print(options)
option = ticker.option_chain(options[3]).calls
print(option)
plt.clf()
plt.plot([elem for elem in options],[len(ticker.option_chain(options[k]).calls) for k in range(len(options))])
plt.xlabel("date d'expiration")
plt.ylabel("nombre de calls")
plt.show()

import yfinance as yf

# Specify the stock ticker symbol
ticker_symbol = 'AAPL'  # Example: Apple Inc.

# Get the stock data
stock = yf.Ticker(ticker_symbol)

# Fetch the dividend data
dividends = stock.dividends

# Print the dividends
print(dividends)
