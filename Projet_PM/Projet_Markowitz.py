import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
input_date = "2025-08-25"
delay = 10*365
input_date = datetime.strptime(input_date, "%Y-%m-%d")
time_period = input_date - timedelta(days=int(delay))

def CAPM(stock): 
    #risk free rate
    treasury_yield = yf.Ticker("^TNX")
    historical_data = treasury_yield.history(start=time_period, end=input_date)
    average_yield = historical_data['Close'].mean()


    #get the yield of the S&P
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(start=time_period, end=input_date)  
    data['Daily Return'] = data['Close'].pct_change()
    total_return = ((1 + data['Daily Return']).prod())**(1/(delay/365))
    total_return = (total_return - 1)*100

    #get the beta of the stock ( could be done manually but i can't manage the get the right value so i'd rather take the one from yfinance.)
    stock = yf.Ticker(stock)
    beta = stock.info['beta']
    CAPM = average_yield + beta*(total_return- average_yield)
    print(beta,average_yield,total_return)
    return CAPM
print(CAPM("MSFT"))
#à faire
#implémenter matrice varianve covariance en utilisant le HULL
def cov_estim(stock1,stock2):
    #estimation of the volatility of a stock. Time period is 100 days before input date
    m=5*1000
    lambd = 0.1
    date_exacte = input_date 

    ticker_symbol1 = stock1  # Example: Apple Inc.
    ticker_symbol2 = stock2

    ticker1 = yf.Ticker(ticker_symbol1)
    ticker2 = yf.Ticker(ticker_symbol2)

    cours_action1 = ticker1.history(period = '5y')
    cours_action2 = ticker2.history(period = '5y')

    cours_fermeture1 = cours_action1["Close"]
    cours_fermeture2 = cours_action2["Close"]
    cov = 0
    a = 0
    for i in range(1,m):
            jour1 = date_exacte - timedelta(days = i)
            jour0 = date_exacte - timedelta(days = i-1)
            
            jour1 = jour1.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')
            jour0 = jour0.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')

            if (jour1 in cours_fermeture1.keys()) and (jour0 in cours_fermeture1.keys()) and (jour1 in cours_fermeture2.keys()) and (jour0 in cours_fermeture2.keys()):
                u1 = (cours_fermeture1[jour1]/cours_fermeture1[jour0] - 1)
                u2 = (cours_fermeture2[jour1]/cours_fermeture2[jour0] - 1)
                cov += u1*u2
                a +=1
    return (cov/a)*252
def cor_estim(stock1,stock2):
     return cov_estim(stock1,stock2)/np.sqrt(cov_estim(stock1,stock1)*cov_estim(stock2,stock2))
def cor_matrix(pf):
     matrix = np.array([[cor_estim(stock1,stock2) for stock2 in pf] for stock1 in pf])
     return matrix
pf = np.array(['AAPL','PLTR'])
Sigma = cor_matrix(pf)
R = np.array([CAPM(stock) for stock in pf])
n = len(pf)
def weights(mu):
        
        # Construct the block matrix and vectors
        A_top = np.block([[2 * Sigma, -R.reshape(-1, 1), -np.ones((n, 1))]])
        A_middle = np.block([[R.reshape(1, -1), np.zeros((1, 1)), np.zeros((1, 1))]])
        A_bottom = np.block([[np.ones((1, n)), np.zeros((1, 1)), np.zeros((1, 1))]])
        A = np.vstack([A_top, A_middle, A_bottom])

        b = np.concatenate([np.zeros(n), [mu], [1]])

        # Solve the system
        solution = np.linalg.solve(A, b)

        w = solution[:n]
        lambda_1 = solution[n]
        lambda_2 = solution[n+1]

        return np.array(w)

plt.plot([weights(mu).T @ cor_matrix(pf) @ weights(mu) for mu in np.linspace(-100,100,30)],[weights(mu).T @ R for mu in np.linspace(-100,100,30)],'bo')
plt.show()