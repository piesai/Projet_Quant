import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import norm
#extraction des données
ticker_symbol = 'AAPL'  # Example: Apple Inc.
ticker = yf.Ticker(ticker_symbol)
cours_action = ticker.history(period = '5y')
cours_fermeture = cours_action["Close"]


dates_expi= ticker.options
option = ticker.option_chain(dates_expi[3]).calls
print(option)

#calcul des écarts entre théorie et pratique avec BSM 

Ecarts = np.array([])

for ind_Tf in range(len(dates_expi)):
    tick = ticker.option_chain(dates_expi[ind_Tf]).calls
    M = []
    for k in range(len(tick)):
        #calcul date_exacte
        date_exacte = tick["lastTradeDate"][k]
        date_exe = pd.Timestamp(dates_expi[ind_Tf],tz = "America/New_York")
        delta = date_exe - date_exacte
        T = (delta.days)*(5/7)*(1/252) #calcul un peu grossier où on prend 5/7 pour prendre en compte les week ends
        

        vol_imp = tick["impliedVolatility"][k] #un peu ironique de calcule BSM avec la vol implicite. On peut estimer la vol avec EWMA (amélioration possible)
        K = tick["strike"][k]
        date_string = date_exacte.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')

        if (date_string in cours_fermeture.keys()) and not(tick["inTheMoney"][k]):
            S = cours_fermeture[date_string]
            r = 0.0425 #taux approximatif de la FED. Améliorations possibles
            #calcul de BSM : 
            d_1 = (np.log(S/K) + (r+(vol_imp**2)/2)*T)/(vol_imp*np.sqrt(T))
            d_2 = d_1 - vol_imp*np.sqrt(T)
            Nd_1 = norm.cdf(d_1)
            Nd_2 = norm.cdf(d_1)
            prix_call = S*Nd_1 - K*np.exp(-r*T)*Nd_2
            e = abs((prix_call-tick["lastPrice"][k])/tick["lastPrice"][k])
            if prix_call < 0:
                print(tick["lastPrice"][k])

            M.append(e)

print(np.average(M))
print(np.std(M))

        