from scipy.stats import norm
import numpy as np

def formulaBS(S0, r, q, vol, T, K):
    K = np.array(K)
    d_1 = (np.log(S0/K)+(r-q+vol**2/2)*T)/(vol*np.sqrt(T))
    d_2 = d_1 - (vol*np.sqrt(T))

    call_option = S0*np.exp(-q*T)*norm.cdf(d_1)-K*np.exp(-r*T)*norm.cdf(d_2)
    return call_option