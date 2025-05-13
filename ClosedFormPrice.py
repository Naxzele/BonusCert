from scipy.stats import norm
import numpy as np

def formulaBS(S0,K,T,vol,r,div_yield):
    d_1 = (np.log(S0/K)+(r-div_yield+vol**2/2)*T)/(vol*np.sqrt(T))
    d_2 = d_1 - (vol*np.sqrt(T))

    call_option = S0*np.exp(-div_yield*T)*norm.cdf(d_1)-K*np.exp(-r*T)*norm.cdf(d_2)
    return call_option