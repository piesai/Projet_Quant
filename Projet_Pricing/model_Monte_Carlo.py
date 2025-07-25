import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta
from scipy.stats import norm
#extraction des données
ticker_symbol = 'AAPL'  # Example: Apple Inc.
ticker = yf.Ticker(ticker_symbol)
cours_action = ticker.history(period = '5y')
cours_fermeture = cours_action["Close"]
dates_expi= ticker.options
options = ticker.options
option = ticker.option_chain(options[3]).calls
r = 0.10
N = 5

def binomial(T,N,K,S,vol):
    u = float(np.exp(vol*np.sqrt(T/N)))
    d = 1/u
    p = float((np.exp(r*T/N) - d)/(u-d))
    f = [[0 for j in range(i+1)] for i in range(0,N+1)]
    for j in range(0,N+1):
        f[N][j] = max(S*(u**j)*(d**(N-j)) - K,0)
    for i in range(0,N):
        i = N - i -1
            
        for j in range(0,i):
    
            f[i][j] = max(S*(u**j)*(d**(i-j)) - K, float(np.exp(-r*(T/N)))*(p*f[i+1][j+1] + (1-p)*f[i+1][j]))
    
    
    f[0][0]  =  max(S - K, float(np.exp(-r*(T/N)))*(p*f[1][1] + (1-p)*f[1][0]))
    print(f)
    return f[0][0]


print(binomial(0.0271,10,212.52,214.5,1.03e-05))
