import QuantLib as ql
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates(m=1)
calculation_date = ql.Date(17, 3, 2026)
spot = 185.7899
ql.Settings.instance().evaluationDate = calculation_date
dividend_yield = ql.QuoteHandle(ql.SimpleQuote(0.0))
risk_free_rate = 0.01
dividend_rate = 0.00
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
import yfinance as yf
import QuantLib as ql
from datetime import datetime

# Define the ticker symbol for Amazon
ticker = "AMZN"

# Create a Ticker object
amzn = yf.Ticker(ticker)

# Get all available expiration dates for options
expirations = amzn.options

# Convert expiration strings to ql.Date objects
ql_expirations = []
for exp_str in expirations:
    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    ql_exp = ql.Date(exp_date.day, exp_date.month, exp_date.year)
    ql_expirations.append(ql_exp)

# Get all unique strike prices from the first expiration's call options
opt = amzn.option_chain(expirations[0])
strikes = sorted(opt.calls['strike'].unique())

# Initialize the data matrix with None
data = [[None for _ in strikes] for _ in ql_expirations]

# Fill the data matrix
for i, exp_str in enumerate(expirations):
    opt = amzn.option_chain(exp_str)
    calls = opt.calls
    for j, strike in enumerate(strikes):
        implied_vol = calls[calls['strike'] == strike]['impliedVolatility']
        if not implied_vol.empty:
            data[i][j] = implied_vol.values[0]

# Interpolate None values with the mean of the two nearest non-None values in the same row
for i in range(len(ql_expirations)):
    row = data[i]
    for j in range(len(strikes)):
        if row[j] is None:
            left = j - 1
            right = j + 1
            # Find the nearest non-None values
            while left >= 0 and row[left] is None:
                left -= 1
            while right < len(strikes) and row[right] is None:
                right += 1
            if left >= 0 and right < len(strikes):
                row[j] = (row[left] + row[right]) / 2
            elif left >= 0:
                row[j] = row[left]
            elif right < len(strikes):
                row[j] = row[right]

# Print the results
print("Expiration Dates (ql.Date):")
for date in ql_expirations:
    print(date)

print("\nStrikes:", strikes)

print("\nData Matrix (Implied Volatility, interpolated):")
for row in data:
    print(row)
expiration_dates = ql_expirations

###################""
implied_vols = ql.Matrix(len(data[0]), len(data))
for i in range(implied_vols.rows()):
    for j in range(implied_vols.columns()):
        implied_vols[i][j] = data[j][i]
black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vols, day_count)

v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; sigma = 0.5;

process = ql.HestonProcess(flat_ts, dividend_ts,
                           ql.QuoteHandle(ql.SimpleQuote(spot)),
                           v0, kappa, theta, sigma, rho)
model = ql.HestonModel(process)
engine = ql.AnalyticHestonEngine(model)
heston_helpers = []
black_var_surface.setInterpolation("bicubic")
one_year_idx = 20 # 12th row in data is for 1 year expiry
date = expiration_dates[one_year_idx]
print(date)
for j, s in enumerate(strikes):
    t = (date - calculation_date )
    p = ql.Period(t, ql.Days)
    sigma = data[one_year_idx][j]
    #sigma = black_var_surface.blackVol(t/365.25, s)
    helper = ql.HestonModelHelper(p, calendar, spot, s,
                                  ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                  flat_ts,
                                  dividend_ts)
    helper.setPricingEngine(engine)
    heston_helpers.append(helper)
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
model.calibrate(heston_helpers, lm,
                 ql.EndCriteria(5000, 500, 1.0e-8,1.0e-8, 1.0e-8))
theta, kappa, sigma, rho, v0 = model.params()

print ("\ntheta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % (theta, kappa, sigma, rho, v0))
avg = 0.0

print ("%15s %15s %15s %20s" % (
    "Strikes", "Market Value",
     "Model Value", "Relative Error (%)"))
print ("="*70)
for i, opt in enumerate(heston_helpers):
    print(opt.marketValue())
    print(opt.modelValue())
    err = (opt.modelValue()/opt.marketValue() - 1.0)
    print ("%15.2f %14.5f %15.5f %20.7f " % (
        strikes[i], opt.marketValue(),
        opt.modelValue(),
        100.0*(opt.modelValue()/opt.marketValue() - 1.0)))
    avg += abs(err)
avg = avg*100.0/len(heston_helpers)
print ("-"*70)
print ("Average Abs Error (%%) : %5.3f" % (avg))