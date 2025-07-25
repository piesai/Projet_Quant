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
r = 0.0425
N = 100

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
    return f[0][0]


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
        T+=0.03
        
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

            
            prix_call = binomial(T,N,K,S,vol)
            e1 = abs((prix_call-tick["lastPrice"][k])/tick["lastPrice"][k])
            
            M1.append(e1)
            if int(100*(S/K)) in dico1.keys():
                    dico1[int(100*(S/K))].append(e1)

            else: 
                dico1[int(100*(S/K))] = [e1]

        #volatilité implicite
        if (date_string in cours_fermeture.keys()):
            vol = tick["impliedVolatility"][k]
            S = cours_fermeture[date_string]

            prix_call =  binomial(T,N,K,S,vol)
            e1 = abs((prix_call-tick["lastPrice"][k])/tick["lastPrice"][k])
            M2.append(e1)
            if int(100*(S/K)) in dico2.keys():
                    dico2[int(100*(S/K))].append(e1)

            else: 
                dico2[int(100*(S/K))] = [e1]
            
print("moyenne de l'erreur : ",np.average(M1))
print("ecart type de l'erreur : ",np.std(M1))
for m in dico1.keys():
    dico1[m] = np.average(dico1[m])

L= list(dico1.keys())
L.sort()
G = [dico1[k] for k in L]
plt.plot(L,G)
plt.plot([100,100],[-10,30],"r")
plt.xlabel("rapport 100*(S/K)")
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
plt.plot([100,100],[-10,30],"r")
plt.xlabel("rapport 100*(S/K)")
plt.ylabel("erreure relative en pourcentage")
plt.title("à gauche de la ligne rouge, out of the money. A droite, in the money. Volatilité utilisée : volatilité implicite")
plt.show()