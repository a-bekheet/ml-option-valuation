'''
We have the following:
1. Stock Price (S) --> stockClose
2. Strike Price (K): strike
3. Time to Expiry (T): daysToExpiry (needs conversion to years: T = daysToExpiry/365)
4. Implied Volatility (σ): impliedVolatility
5. Option Type: From contractSymbol ('C' or 'P')

We need:
1. Risk Free Interest Rate (r)
params:
a. date
b. currency

'''