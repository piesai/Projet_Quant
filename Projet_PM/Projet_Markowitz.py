import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
input_date = "2025-08-12"
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
print(CAPM("AAPL"))