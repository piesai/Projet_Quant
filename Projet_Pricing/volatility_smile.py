import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#extraction des données
ticker_symbol = 'AAPL'  # Example: Apple Inc.
ticker = yf.Ticker(ticker_symbol)
cours_action = ticker.history(period = '5y')
cours_fermeture = cours_action["Close"]
dates_expi= ticker.options
options = ticker.options


#calcul des écarts entre théorie et pratique avec BSM 
Vol_imp = {}
Vol_estim = {}

for ind_Tf in range(len(dates_expi)):
    tick = ticker.option_chain(dates_expi[ind_Tf]).calls
    for k in range(len(tick)):
        #calcul date_exacte
        date_exacte = tick["lastTradeDate"][k]
        date_exe = pd.Timestamp(dates_expi[ind_Tf],tz = "America/New_York")
        delta = date_exe - date_exacte
        T = (delta.days)*(5/7)*(1/252) #calcul un peu grossier où on prend 5/7 pour prendre en compte les week ends
        date_string = date_exacte.strftime('%Y-%m-%d' +  ' 00:00:00-04:00')
        T += 0.01 # pour que ça marche le vendredi :')

        
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
            S = cours_fermeture[date_string]
            if (int(100*(K/S)),T) in Vol_estim.keys():
                    Vol_estim[(int(100*(K/S)),T)].append(vol)

            else: 
                Vol_estim[(int(100*(K/S)),T)] = [vol]

        #volatilité implicite
        if (date_string in cours_fermeture.keys()):
            vol = tick["impliedVolatility"][k]
            S = cours_fermeture[date_string]
            if (int(100*(K/S)),T) in Vol_imp.keys():
                    Vol_imp[(int(100*(K/S)),T)].append(vol)

            else: 
                Vol_imp[(int(100*(K/S)),T)] = [vol]
            

for m in Vol_estim.keys():
    Vol_estim[m] = np.average(Vol_estim[m])
D = Vol_estim
keys = np.array(list(D.keys()))
values = np.array(list(D.values()))
x_coords = keys[:, 0]
y_coords = keys[:, 1]
z_coords = values
rbf = Rbf(x_coords, y_coords, z_coords, function='linear')
x_grid, y_grid = np.meshgrid(np.linspace(min(x_coords), max(x_coords), 100),
                             np.linspace(min(y_coords), max(y_coords), 100))
z_grid = rbf(x_grid, y_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
ax.set_xlabel('K/S')
ax.set_ylabel('T')
ax.set_zlabel('Vol')
plt.show()


D = Vol_imp
keys = np.array(list(D.keys()))
values = np.array(list(D.values()))
x_coords = keys[:, 0]
y_coords = keys[:, 1]
z_coords = values
rbf = Rbf(x_coords, y_coords, z_coords, function='linear')
x_grid, y_grid = np.meshgrid(np.linspace(min(x_coords), max(x_coords), 100),
                             np.linspace(min(y_coords), max(y_coords), 100))
z_grid = rbf(x_grid, y_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
ax.set_xlabel('K/S')
ax.set_ylabel('T')
ax.set_zlabel('Vol')
plt.show()

