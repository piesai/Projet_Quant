import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta
from scipy.stats import norm
import math
#extraction des données
ticker_symbol = 'AAPL'  # Example: Apple Inc.
ticker = yf.Ticker(ticker_symbol)
cours_action = ticker.history(period = '5y')
cours_fermeture = cours_action["Close"]
dates_expi= ticker.options
options = ticker.options
#calcul des écarts entre théorie et pratique avec BSM 
M1=[]
M2 = []
dico1 = {}
dico2 = {}

for ind_Tf in range(len(dates_expi)):
    tick = ticker.option_chain(dates_expi[ind_Tf]).calls
    for k in range(len(tick)):
        #calcul date_exacte
        date_exacte = tick["lastTradeDate"][k]
        date_exe = pd.Timestamp(dates_expi[ind_Tf],tz = "America/New_York")
        delta = date_exe - date_exacte
        T = (delta.days)*(5/7)*(1/252) #calcul un peu grossier où on prend 5/7 pour prendre en compte les week ends
        date_string = date_exacte.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')

        
        #estimateur sans biais de la volatilité
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
        #strike price
        K = tick["strike"][k]


        #volatilité estimée
        if (date_string in cours_fermeture.keys()):
            r = 0.0425 #taux approximatif de la FED. Améliorations possibles
            S = cours_fermeture[date_string]

            #calcul de BSM : 
            
            d_1 = (np.log(S/K) + (r+(vol**2)/2)*T)/(vol*np.sqrt(T))
            d_2 = d_1 - vol*np.sqrt(T)
            Nd_1 = norm.cdf(d_1)
            Nd_2 = norm.cdf(d_2)
            prix_call = S*Nd_1 - K*np.exp(-r*T)*Nd_2
            e1 = abs((prix_call-tick["ask"][k])/tick["ask"][k])
            if not math.isinf(e1) and not math.isnan(tick["volume"][k]):
                for l in range(int(tick["volume"][k])):
                     M1.append(e1)
                if int(100*(K/S)) in dico1.keys():
                        dico1[int(100*(K/S))].append(e1)
                else: 
                    dico1[int(100*(K/S))] = [e1]
                

        #volatilité implicite
        if (date_string in cours_fermeture.keys()):
            vol = tick["impliedVolatility"][k]
            
            r = 0.0425 #taux approximatif de la FED. Améliorations possibles
            S = cours_fermeture[date_string]
            #calcul de BSM : s
            d_1 = (np.log(S/K) + (r+(vol**2)/2)*T)/(vol*np.sqrt(T))
            d_2 = d_1 - vol*np.sqrt(T)
            Nd_1 = norm.cdf(d_1)
            Nd_2 = norm.cdf(d_2)
            prix_call = S*Nd_1 - K*np.exp(-r*T)*Nd_2
            e1 = abs((prix_call-tick["ask"][k])/tick["ask"][k])
        
            if not math.isinf(e1) and not math.isnan(tick["volume"][k]):
                for l in range(int(tick["volume"][k])):
                     M2.append(e1)
                if int(100*(K/S)) in dico2.keys():
                        dico2[int(100*(K/S))].append(e1)
                else: 
                    dico2[int(100*(K/S))] = [e1]

            
print("moyenne de l'erreur : ",np.average(M1))
print("ecart type de l'erreur : ",np.std(M1))
for m in dico1.keys():
    dico1[m] = np.average(dico1[m])

L= list(dico1.keys())
L.sort()
G = [dico1[k] for k in L]
plt.plot(L,G)
plt.plot([100,100],[-1,1],"r")
plt.xlabel("rapport 100*(K/S)")
plt.ylabel("erreure relative en pourcentage")
plt.title("à gauche de la ligne rouge, out of the money. A droite, in the money. Volatilitée utilisée : volatilité estimée à partir du cours de l'action")
plt.show()

print("moyenne de l'erreur : ",np.average(M2))
print("ecart type de l'erreur : ",np.std(M2))
for m in dico2.keys():
    dico2[m] = np.average(dico2[m])

L= list(dico2.keys())
L.sort()
G = [dico2[k] for k in L]
plt.plot(L,G)
plt.plot([100,100],[-1,1],"r")
plt.xlabel("rapport 100*(K/S)")
plt.ylabel("erreure relative en pourcentage")
plt.title("à gauche de la ligne rouge, out of the money. A droite, in the money. Volatilité utilisée : volatilité implicite")
plt.show()