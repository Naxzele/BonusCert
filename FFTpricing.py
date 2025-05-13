import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d
import pandas as pd

# Characteristic function for Black-Scholes model
def bs_characteristic_fn(u, S0, r, q, T, params):
    sigma = params['sigma']
    u = u[:, np.newaxis]  # Shape (N, 1)
    r = np.asarray(r)[np.newaxis, :]  # Shape (1, M)
    q = np.asarray(q)[np.newaxis, :]
    T = np.asarray(T)[np.newaxis, :]

    log_S0 = np.log(S0)
    drift = (r - q - 0.5 * sigma**2) * T
    exponent = 1j * u * (log_S0 + drift) - 0.5 * sigma**2 * (u**2) * T
    return np.exp(exponent)  # Shape (N, M)

def bates_characteristic_fn(u, S0, r, q, T, params):
    """
    - kappa: mean reversion speed of variance
    - theta: long-run variance
    - sigma: vol of vol
    - rho: correlation between asset and variance
    - v0: initial variance
    - lambda: jump intensity (expected # jumps per year)
    - muJ: mean of log jump size
    - sigmaJ: std dev of log jump size
    """

    # Unpack parameters
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho   = params['rho']
    v0    = params['v0']
    lambda_ = params['lambda']
    mu_J  = params['muJ']
    sigma_J = params['sigmaJ']

    u = u[:, np.newaxis]  # Shape (N, 1)
    r = np.asarray(r)[np.newaxis, :]  # Shape (1, M)
    q = np.asarray(q)[np.newaxis, :]
    T = np.asarray(T)[np.newaxis, :]

    i = 1j
    d = np.sqrt((rho * sigma_v * i * u - kappa)**2 + (sigma_v**2) * (i * u + u**2))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)

    exp_dt = np.exp(-d * T)
    G = g * exp_dt
    C = (r - q) * i * u * T + lambda_ * T * (np.exp(i * u * mu_J - 0.5 * sigma_J**2 * u**2) - 1)
    C += theta * kappa / (sigma_v**2) * ((kappa - rho * sigma_v * i * u - d) * T - 2 * np.log((1 - G) / (1 - g)))

    D = (kappa - rho * sigma_v * i * u - d) / (sigma_v**2) * ((1 - exp_dt) / (1 - G))

    return np.exp(C + D * v0 + i * u * np.log(S0))

class fft_price:
    def __init__(self, char_fn, S0, r, q, T, params):
        self.S0 = S0
        self.r = np.array(r)
        self.q = np.array(q)
        self.T = np.array(T)

        if char_fn == 'BSM':
            self.char_fn = lambda u: bs_characteristic_fn(u, S0, self.r, self.q, self.T, params)

        if char_fn == 'BATES':
            self.char_fn = lambda u: bates_characteristic_fn(u, S0, self.r, self.q, self.T, params)

    def prices(self, alpha=1.5, N=4096, eta=0.25, strike=None):
        lambd = 2 * np.pi / (N * eta)
        b = 0.5 * N * lambd
        k = -b + lambd * np.arange(N)  # shape (N,)
        K = np.exp(k)

        v = eta * np.arange(N)  # shape (N,)
        i = 1j

        # Simpson's rule weights (shape N,)
        w = np.ones(N)
        w[0] = w[-1] = 1 / 3
        w[1:-1:2] = 4 / 3
        w[2:-1:2] = 2 / 3
        w = w[:, None]  # Shape (N,1) for broadcasting

        # Matrix characteristic function
        phi_call = self.char_fn(v - (alpha + 1) * i)  # Shape (N, M)
        discount = np.exp(-self.r * self.T)[None, :]  # Shape (1, M)

        # Call integrand (N, M)
        denominator_call = (alpha**2 + alpha - v[:, None]**2 + i * (2 * alpha + 1) * v[:, None])
        integrand_call = discount * phi_call / denominator_call * np.exp(i * v[:, None] * b) * eta * w

        # FFT for all columns
        fft_call = np.real(fft(integrand_call, axis=0))
        call_prices = np.exp(-alpha * k[:, None]) / np.pi * fft_call  # shape (N, M)

        # Repeat for puts
        # phi_put = self.char_fn(v - (alpha - 1) * i)
        # denominator_put = (alpha**2 - alpha - v[:, None]**2 + i * (2 * alpha - 1) * v[:, None])
        # integrand_put = discount * phi_put / denominator_put * np.exp(i * v[:, None] * b) * eta * w
        # fft_put = np.real(fft(integrand_put, axis=0))
        # put_prices = np.exp(-alpha * k[:, None]) / np.pi * fft_put  # shape (N, M)

        if strike is not None:
            strike = np.asarray(strike)
            results_call = []
            # results_put = []

            for m in range(call_prices.shape[1]):
                call_interp = interp1d(K, call_prices[:, m], kind='cubic', fill_value="extrapolate")
                # put_interp = interp1d(K, put_prices[:, m], kind='cubic', fill_value="extrapolate")
                results_call.append(call_interp(strike))
                # results_put.append(put_interp(strike))

            calls = np.stack(results_call, axis=-1)
            # puts = np.stack(results_put, axis=-1)

            result_df = pd.DataFrame({
                'K': np.repeat(strike, calls.shape[1]),
                'T': np.tile(self.T, calls.shape[0]),
                'q': np.tile(self.q, calls.shape[0]) if self.q.shape[0]>1 else np.repeat(self.q, calls.shape[0]*calls.shape[1]),
                'r': np.tile(self.r, calls.shape[0]) if self.r.shape[0]>1 else np.repeat(self.r, calls.shape[0]*calls.shape[1]),
                'calls': calls.ravel()
            })

            result_df['puts'] = result_df['calls'] - self.S0 * np.exp(-result_df['q'] * result_df['T']) + result_df['K'] * np.exp(-result_df['r'] * result_df['T'])

            return result_df
        else:
            return K, call_prices  # Shape (N, M)