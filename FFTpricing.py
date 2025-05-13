import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 1. Characteristic function for Black-Scholes model
def bs_characteristic_fn(u, S0, r,q, sigma, T):
    return np.exp(1j * u * (np.log(S0) + (r - q -0.5 * sigma**2) * T) - 0.5 * sigma**2 * u**2 * T)

# 2. FFT pricing using Carr-Madan method
def fft_call_price_bs(S0, r, q, sigma, T, alpha=1.5, N=4096, eta=0.25, strike=None):
    # Step sizes and grids
    lambd = 2 * np.pi / (N * eta)
    b = 0.5 * N * lambd
    k = -b + lambd * np.arange(N)  # log-strike grid
    K = np.exp(k)                  # strike grid

    v = eta * np.arange(N)        # integration grid in Fourier domain
    i = complex(0.0, 1.0)

    # Simpson's rule weights
    w = np.ones(N)
    w[0] = w[-1] = 1.0 / 3
    w[1:-1:2] = 4.0 / 3
    w[2:-1:2] = 2.0 / 3

    # Carr-Madan integrand
    phi = bs_characteristic_fn(v - (alpha + 1) * i, S0, r, q, sigma, T)
    numerator = np.exp(-r * T) * phi
    denominator = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    integrand = numerator / denominator * np.exp(i * v * b) * eta * w

    # FFT
    fft_values = np.real(fft(integrand))
    call_prices = np.exp(-alpha * k) / np.pi * fft_values

    if strike is not None:
        # Interpolate to get price at a specific strike
        interpolator = interp1d(K, call_prices, kind='cubic', fill_value="extrapolate")
        return interpolator(strike)
    else:
        return K, call_prices