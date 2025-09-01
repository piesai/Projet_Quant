import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
input_date = "2021-08-25"
delay = 5*365
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
    if a < 10: #on évite des trop petits samplse
         return 999
    return (cov/a)*252
def cor_estim(stock1,stock2):
     return cov_estim(stock1,stock2)/np.sqrt(cov_estim(stock1,stock1)*cov_estim(stock2,stock2))
def cor_matrix(pf):
     matrix = []
     for k in range(len(pf)):
          print(k)
          L = [cov_estim(pf[k],pf[j]) for j in range(len(pf))]
          if not(999 in L):
               matrix.append(L)
     return np.array(matrix)
pf = np.array(['AAPL','LVMHF','PLTR','AIR','GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY'])
Sigma = cor_matrix(pf)
R = np.array([CAPM(stock) for stock in pf])
n = len(pf)
# Construct the block matrix and vectors
A_top = np.block([[2 * Sigma, -R.reshape(-1, 1), -np.ones((n, 1))]])
A_middle = np.block([[R.reshape(1, -1), np.zeros((1, 1)), np.zeros((1, 1))]])
A_bottom = np.block([[np.ones((1, n)), np.zeros((1, 1)), np.zeros((1, 1))]])
A = np.vstack([A_top, A_middle, A_bottom])
def weights(mu):
        

        b = np.concatenate([np.zeros(n), [mu], [1]])
                                           
        # Solve the system
        solution = np.linalg.solve(A, b)
        w = solution[:n]
        lambda_1 = solution[n]
        lambda_2 = solution[n+1]

        return np.array(w)
def var_return(min,max,i):
     L = []
     M = []
     for mu in np.linspace(min,max,i):
          w = weights(mu)
          L.append( w.T @ Sigma @ w )
          M.append( w.T @ R)
     return L,M
L,M = var_return(-100,100,30)
plt.plot(L,M,'bo')
plt.show()
def pf_visu(mu,pf):
    date_exacte = input_date 

    #estimation of the volatility of a stock. Time period is 100 days before input date
    w = weights(mu)

    #définition des tickers
    tickers = []
    for stock in pf:
         tickers.append(yf.Ticker(stock))
    
    #cours des actions
    cours_actions = []
    for ticker in tickers:
         cours_action = ticker.history(period = '5y')
         cours_actions.append(cours_action["Close"])
    
    cours_final = []
    S_P =  yf.Ticker("^GSPC").history(period='5y')["Close"]
    SP = []
    temps = []
    a = 0
    while a < 5*365:
            
            jour_exacte = date_exacte + timedelta(days = a)
            jour1 = jour_exacte.strftime('%Y-%m-%d' +  ' 01:00:00-04:00')
            jour0 = jour_exacte.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')
            if (jour1 in cours_actions[1].keys()):
                if (jour1 in S_P.keys()): 
                    cours_final.append(np.dot(w,[cours_actions[k][jour1] for k in range(len(pf))]))
                    temps.append(jour1)
                    SP.append(S_P[jour1]/10)
                elif jour0 in S_P.keys():
                    cours_final.append(np.dot(w,[cours_actions[k][jour1] for k in range(len(pf))]))
                    temps.append(jour1)
                    SP.append(S_P[jour0]/10)
            elif jour0 in cours_actions[1].keys():
                if (jour0 in S_P.keys()): 
                    cours_final.append(np.dot(w,[cours_actions[k][jour0] for k in range(len(pf))]))
                    temps.append(jour0)
                    SP.append(S_P[jour0]/10)
                elif jour1 in S_P.keys():
                    cours_final.append(np.dot(w,[cours_actions[k][jour0] for k in range(len(pf))]))
                    temps.append(jour0)
                    SP.append(S_P[jour1]/10)
            
            a+=1
    return cours_final,temps,SP

import numpy as np


def Sharpe(SP,portfolio_values, risk_free_rate):
    """
    Calculate the Sharpe ratio for a portfolio.

    Parameters:
    - portfolio_values: List of portfolio values over time
    - risk_free_rate: Annualized risk-free rate (as a decimal, e.g., 0.02 for 2%)

    Returns:
    - Sharpe ratio
    """
    # Calculate daily returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Annualize the risk-free rate for daily data
    if len(portfolio_values) > 30:  # Assume daily data if more than 30 points
        risk_free_rate_daily = (1 + risk_free_rate) ** (1/252) - 1
    else:  # Assume monthly data
        risk_free_rate_daily = (1 + risk_free_rate) ** (1/12) - 1

    # Calculate excess returns
    excess_returns = returns - risk_free_rate_daily

    # Sharpe ratio
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    # Annualize Sharpe ratio
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(len(returns))

    return sharpe_ratio_annualized

cours_final1,temps1,SP= pf_visu(9.1,pf)
print(Sharpe(SP,cours_final1,0.04))
plt.plot(temps1,cours_final1, linestyle='-', color='g')
cours_final2,temps2,SP= pf_visu(-50,pf)
plt.plot(temps2,cours_final2, linestyle='-', color='c')
cours_final3,temps3,SP= pf_visu(50,pf)
print(Sharpe(SP,cours_final3,0.04))
plt.plot(temps3,cours_final3, linestyle='-', color='b')
print(Sharpe(SP,SP,0.04))
#A FAIRE PROCHAINE FOIS : 
     #-implémenter une mega matrice de cov 500x500 ?
     #-vérifier si les valeurs attendues correspondent à MPT
plt.plot(temps3,SP, linestyle='-', color='r')

plt.show()

