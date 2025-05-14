import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d
import pandas as pd
from scipy.optimize import minimize

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

    def prices(self, strike, alpha=1.5, N=4096, eta=0.25):
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

        strike = np.asarray(strike)
        results_call = []

        for m in range(call_prices.shape[1]):
            call_interp = interp1d(K, call_prices[:, m], kind='cubic', fill_value="extrapolate")
            results_call.append(call_interp(strike))

        calls = np.stack(results_call, axis=-1)
        result_df = pd.DataFrame({
            'K': np.repeat(strike, calls.shape[1]),
            'T': np.tile(self.T, calls.shape[0]),
            'q': np.tile(self.q, calls.shape[0]) if self.q.shape[0]>1 else np.repeat(self.q, calls.shape[0]*calls.shape[1]),
            'r': np.tile(self.r, calls.shape[0]) if self.r.shape[0]>1 else np.repeat(self.r, calls.shape[0]*calls.shape[1]),
            'calls': calls.ravel()
        })

        result_df['puts'] = result_df['calls'] - self.S0 * np.exp(-result_df['q'] * result_df['T']) + result_df['K'] * np.exp(-result_df['r'] * result_df['T'])

        return result_df
    
class ModelCalibrator:
    def __init__(self, model_type, S0, r_term, q_term,  calls_data, puts_data, params, bounds=None, constraints=None):
        """
        Initialize the calibrator with model type (BSM or BATES)
        """
        self.model_type = model_type
        self.S0 = S0
        self.params = params
        self.bounds = bounds
        self.constraints = constraints

        # Extract market data
        self.strikes = np.unique(np.concatenate((calls_data['K'].unique(),puts_data['K'].unique())))
        self.T = np.unique(np.concatenate((calls_data['T'].unique(),puts_data['T'].unique())))
        self.market_calls = calls_data.sort_values(['K','T'])['calls'].values
        self.market_puts = puts_data.sort_values(['K','T'])['puts'].values

        self.keys_calls = calls_data['K'].round(3).astype(str) + "_" + calls_data['T'].round(6).astype(str)
        self.keys_puts = puts_data['K'].round(3).astype(str) + "_" + puts_data['T'].round(6).astype(str)

        r_interp = interp1d(r_term['T'], r_term['r'], kind='linear', fill_value="extrapolate")
        self.r = r_interp(self.T)
        q_interp = interp1d(q_term['T'], q_term['q'], kind='linear', fill_value="extrapolate")
        self.q = q_interp(self.T)
        
    def set_initial_params(self, params):
        """Set or change parameter values for calibration"""
        self.params = params
        
    def set_bounds(self, bounds):
        """Set or change parameter bounds for optimization"""
        self.bounds = bounds
        
    def set_constraints(self, constraints):
        """Set or change parameter constraints for optimization"""
        self.constraints = constraints
        
    # def _create_price_calculator(self, S0, r, q, T, params):
    #     """Create price calculator instance with current parameters"""
    #     return fft_price(self.model_type, S0, r, q, T, params)
        
    def calculate_prices(self, S0=None, r=None, q=None, T=None, strike=None, params=None):
        """
        Calculate option prices using current parameters
        """
        if S0 is None:
            S0 = self.S0
        
        if r is None:
            r = self.r

        if q is None:
            q = self.q

        if T is None:
            T = self.T

        if strike is None:
            strike = self.strikes

        if params is None:
            params = self.params
            
        calculator = fft_price(self.model_type, S0, r, q, T, params)
        return calculator.prices(strike)

    def error_function(self, params_dict=None, weights=None):
        """
        Calculate error between model and market prices
        """
        # # Convert params_dict with appropriate sign constraints
        # params = {}
        # for k, v in params_dict.items():
        #     # Parameters that must be positive (use exponential to ensure positivity)
        #     if k in ['kappa', 'theta', 'sigma_v', 'v0', 'lambda', 'sigmaJ']:
        #         params[k] = np.exp(v)  # Using log-params for positive-only variables
        #     # Parameters that can be positive or negative
        #     else:
        #         params[k] = v
                
        if params_dict is None:
            params_dict=self.params
        
        # Calculate model prices
        calculator = fft_price(self.model_type, self.S0, self.r, self.q, self.T, params_dict)
        model_prices = calculator.prices(self.strikes)
        model_prices['key'] = model_prices['K'].round(3).astype(str) + "_" + model_prices['T'].round(6).astype(str)
        
        # Calculate errors
        call_errors = model_prices[model_prices['key'].isin(self.keys_calls)]['calls'].values - self.market_calls
        put_errors = model_prices[model_prices['key'].isin(self.keys_puts)]['puts'].values - self.market_puts
        
        # Combine errors
        total_errors = np.concatenate([call_errors, put_errors])
        
        if weights is not None:
            total_errors = total_errors * weights
            
        return np.mean(total_errors**2)
        
    def calibrate(self, positive_bounds ,weights=None, method='L-BFGS-B', options=None):
        """
        Calibrate model parameters to market data with proper sign handling
        """
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-8}

        param_names = list(self.params.keys())
        
        # Transform initial values - take log of positive-only parameters
        initial_values = []
        for name in param_names:
            if name in positive_bounds:
                initial_values.append(np.log(self.params[name]))
            else:
                initial_values.append(self.params[name])
        initial_values = np.array(initial_values)
        
        # Adjust bounds for transformed parameters
        if self.bounds is not None:
            bounds = []
            for name in param_names:
                if name in positive_bounds:
                    # Convert bounds to log-scale
                    lb = np.log(self.bounds[name][0]) if name in self.bounds.keys() else None
                    ub = np.log(self.bounds[name][1]) if name in self.bounds.keys() else None
                    bounds.append((lb, ub))
                else:
                    bounds.append(self.bounds[name]) if name in self.bounds.keys() else bounds.append((None,None))
        else:
            bounds = None
            
        # Optimization function
        def objective(x):
            params_dict = {}
            for i, name in enumerate(param_names):
                if name in positive_bounds:
                    params_dict[name] = np.exp(x[i])  # Transform back
                else:
                    params_dict[name] = x[i]
            return self.error_function(params_dict, weights)
            
        # Run optimization
        result = minimize(objective, 
                        initial_values, 
                        method=method,
                        bounds=bounds,
                        constraints=self.constraints,
                        options=options)
                        
        # Convert back optimized parameters
        optimized_params = {}
        for i, name in enumerate(param_names):
            if name in positive_bounds:
                optimized_params[name] = np.exp(result.x[i])
            else:
                optimized_params[name] = result.x[i]
                
        self.params = optimized_params
        
        return result, optimized_params