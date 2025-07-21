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
print(dates_expi[10])
option = ticker.option_chain(dates_expi[10]).calls
print(option)

#calcul des écarts entre théorie et pratique avec BSM 

Ecarts = np.array([])
M=[]
dico = {}

for ind_Tf in range(len(dates_expi)):
    tick = ticker.option_chain(dates_expi[ind_Tf]).calls
    for k in range(len(tick)):
        #calcul date_exacte
        date_exacte = tick["lastTradeDate"][k]
        date_exe = pd.Timestamp(dates_expi[ind_Tf],tz = "America/New_York")
        delta = date_exe - date_exacte
        T = (delta.days)*(5/7)*(1/252) #calcul un peu grossier où on prend 5/7 pour prendre en compte les week ends
        date_string = date_exacte.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')

        
        #calcul de la volatilité avec EWMA
        m = 200
        Lambda = 0.94
        vol = 0
        for i in range(1,m):
            jour1 = date_exacte - timedelta(days = i)
            jour0 = date_exacte - timedelta(days = i-1)
            
            jour1 = jour1.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')
            jour0 = jour0.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')

            
            if (jour1 in cours_fermeture.keys()) and (jour0 in cours_fermeture.keys()):
                u = (cours_fermeture[jour1]/cours_fermeture[jour0] - 1)
                vol += (u**2)
        #update de la vol
        vol = tick["impliedVolatility"][k]
        #strike price
        K = tick["strike"][k]

        if (date_string in cours_fermeture.keys()):
            r = 0.0425 #taux approximatif de la FED. Améliorations possibles
            S = cours_fermeture[date_string]

            #calcul de BSM : 
            d_1 = (np.log(S/K) + (r+(vol**2)/2)*T)/(vol*np.sqrt(T))
            d_2 = d_1 - vol*np.sqrt(T)
            Nd_1 = norm.cdf(d_1)
            Nd_2 = norm.cdf(d_2)
            prix_call = S*Nd_1 - K*np.exp(-r*T)*Nd_2
            e = abs((prix_call-tick["lastPrice"][k])/tick["lastPrice"][k])
                
            M.append(e)
            if int(100*(S/K)) in dico.keys():
                    dico[int(100*(S/K))].append(e)

            else: 
                dico[int(100*(S/K))] = [e]
            
print(np.average(M))
print(np.std(M))
for m in dico.keys():
    dico[m] = np.average(dico[m])

L= dico.keys()
G = dico.values()
plt.plot(L,G)
plt.show()