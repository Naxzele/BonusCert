import numpy as np
from scipy.interpolate import interp1d

def make_rt(r, dt,T):
    # r(t) component
    maturities = np.arange(0,T+dt,dt)
    r_interp = interp1d(r['T'], r['r'], kind='cubic', fill_value="extrapolate")
    rt = r_interp(maturities)
    rt = (rt[1:]*maturities[1:]-rt[:-1]*maturities[:-1])/(maturities[1:]-maturities[:-1])
    rt = rt[:,np.newaxis]
    return rt


def BSdynamics(n_steps, n_paths, S0, r, q, T, params):
    dt = 1/n_steps
    sigma = params['sigma']
    # Time grid
    t = np.linspace(0, T, n_steps + 1)  # [0, dt, 2dt, ..., T]

    total_steps = n_steps*T
    # r(t) component
    rt = make_rt(r, dt,T)

    # Generate random Brownian increments (Z ~ N(0, 1))
    Z = np.random.standard_normal((total_steps, n_paths))

    # Compute cumulative returns
    drift = (rt - q -0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    returns = np.exp(drift + diffusion * Z)

    # Compute paths: S(t) = S0 * cumulative product of returns
    stockpaths = S0 * np.cumprod(returns, axis=0)
    stockpaths = np.insert(stockpaths, 0, S0, axis=0)
    return stockpaths, returns

def Jump_dynamics(n_steps, n_paths, S0, r, q, T, params):
    # Jump parameters
    sigma = params['sigma']

    lambda_ = params['lambda']
    mu_J  = params['muJ']
    sigma_J = params['sigmaJ'] 

    dt = 1/n_steps
    total_steps = n_steps*T
    # r(t) component
    rt = make_rt(r, dt,T)

    Z = np.random.standard_normal((total_steps, n_paths))
    
    # BS part
    drift = (rt - q - lambda_ * (np.exp(mu_J + 0.5 * sigma_J**2) - 1) -0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Jump part
    jump_shocks = np.random.poisson(lambda_ * dt, (total_steps, n_paths))
    jump_sizes = np.exp(mu_J + sigma_J * np.random.standard_normal((total_steps, n_paths)))

    # Combine diffusion and jumps
    returns = np.exp(drift + diffusion * Z) * (1 + jump_shocks * (jump_sizes - 1))
    stockpaths = S0 * np.cumprod(returns, axis=0)
    stockpaths = np.insert(stockpaths, 0, S0, axis=0)
    return stockpaths, returns

def Heston_dynamics(n_steps, n_paths, S0, r, q, T, params, dismethod):
    # Heston parameters
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho   = params['rho']
    v0    = params['v0']

    dt = 1/n_steps
    total_steps = n_steps*T
    # r(t) component
    rt = make_rt(r, dt,T)

    Z = np.random.standard_normal((total_steps, n_paths))
    # Generate correlated Brownian motions
    Z_v = rho * Z + np.sqrt(1 - rho**2) * np.random.standard_normal((total_steps, n_paths))

    # Simulate volatility paths
    v = np.zeros((total_steps + 1, n_paths))
    v[0] = v0

    if dismethod=='Euler':
        for t in range(1, total_steps):
            v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + 
                sigma_v * np.sqrt(v[t-1] * dt) * Z_v[t-1], 0)  # Ensure v > 0
            
    if dismethod=='Milstein':
        for t in range(1, total_steps):
            v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + 
                sigma_v * np.sqrt(v[t-1] * dt) * Z_v[t-1] + sigma_v**2*dt/4*(Z_v[t-1]**2-1), 0) 

    
    # Simulate S(t) with stochastic vol
    returns = np.exp((rt - q - 0.5 * v) * dt + np.sqrt(v * dt) * Z)
    stockpaths = S0 * np.cumprod(returns, axis=0)
    stockpaths = np.insert(stockpaths, 0, S0, axis=0)
    return stockpaths, returns

def Bates_dynamics(n_steps, n_paths, S0, r, q, T, params, dismethod):
    # Heston parameters
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho   = params['rho']
    v0    = params['v0']

    # jump parameters
    lambda_ = params['lambda']
    mu_J  = params['muJ']
    sigma_J = params['sigmaJ'] 


    dt = 1/n_steps
    total_steps = n_steps*T

    # r(t) component
    rt = make_rt(r, dt,T)

    Z = np.random.standard_normal((total_steps, n_paths))
    # Generate correlated Brownian motions
    Z_v = rho * Z + np.sqrt(1 - rho**2) * np.random.standard_normal((total_steps, n_paths))

    # Simulate volatility paths
    v = np.zeros((total_steps + 1, n_paths))
    v[0] = v0

    if dismethod=='Euler':
        for t in range(1, total_steps):
            v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + 
                sigma_v * np.sqrt(v[t-1] * dt) * Z_v[t-1], 0)  # Ensure v > 0
            
    if dismethod=='Milstein':
        for t in range(1, total_steps):
            v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + 
                sigma_v * np.sqrt(v[t-1] * dt) * Z_v[t-1] + sigma_v**2*dt/4*(Z_v[t-1]**2-1), 0) 
        
    # Jump part
    jump_shocks = np.random.poisson(lambda_ * dt, (total_steps, n_paths))
    jump_sizes = np.exp(mu_J + sigma_J * np.random.standard_normal((total_steps, n_paths)))

    # Simulate S(t) with stochastic vol
    returns = np.exp((rt - q - lambda_ * (np.exp(mu_J + 0.5 * sigma_J**2) - 1) - 0.5 * v[:-1]) * dt + np.sqrt(v[:-1] * dt) * Z) * (1 + jump_shocks * (jump_sizes - 1))

    stockpaths = S0 * np.cumprod(returns, axis=0)
    stockpaths = np.insert(stockpaths, 0, S0, axis=0)
    return stockpaths, returns
                
                
class MCsim:
    def __init__(self, model, n_steps, n_paths, S0, T, params, dismethod='Euler'):
        """
        - model: BSM, BATES
        - dismethod: Euler, Milstein
        - n_steps: number of discrete steps in 1 year
        - n_sim: number of paths to simulate
        - S0: initial value
        - params: parameters of the model
        """
        self.model = model
        self.dismethod = dismethod
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.S0 = S0
        self.T = T
        self.params = params

    def simulate(self, r , q, model=None, dismethod=None, params=None):
        if model is None:
            model = self.model
        if dismethod is None:
            dismethod = self.dismethod
        if params is None:
            params = self.params
        
        if model == 'Bates':
            self.stockpaths, self.returns = Bates_dynamics(self.n_steps, self.n_paths, self.S0, r, q, self.T, params, dismethod)
        if model == 'Jump':
            self.stockpaths, self.returns = Jump_dynamics(self.n_steps, self.n_paths, self.S0, r, q, self.T, params)
        if model == 'Heston':
            self.stockpaths, self.returns = Bates_dynamics(self.n_steps, self.n_paths, self.S0, r, q, self.T, params, dismethod)
        if model == 'BSM':
            self.stockpaths, self.returns =  Bates_dynamics(self.n_steps, self.n_paths, self.S0, r, q, self.T, params)

        return print('Simulation is finished')

    def change_S0(self, S0):
        self.stockpaths = S0 * np.cumprod(self.returns, axis=0)
        self.stockpaths = np.insert(self.stockpaths, 0, S0, axis=0)


    def vanilla_price_batch(self, r, KT_df, stockpaths):
        vanilla_prices = KT_df.copy()

        r_interp = interp1d(r['T'], r['r'], kind='cubic', fill_value="extrapolate")
        vanilla_prices['r'] = r_interp(vanilla_prices['T'])

        strikes_array = np.array(vanilla_prices['K'])[:, np.newaxis]
        ST_array = stockpaths[np.round(vanilla_prices['T']*self.n_steps,0).astype('int')]

        call_payoff = np.maximum(ST_array-strikes_array,0)
        put_payoff = np.maximum(strikes_array-ST_array,0)

        vanilla_prices['calls'] = np.exp(-vanilla_prices['T']*vanilla_prices['r'])*np.mean(call_payoff, axis =1)
        vanilla_prices['puts'] = np.exp(-vanilla_prices['T']*vanilla_prices['r'])*np.mean(put_payoff, axis =1)

        return vanilla_prices

    def down_barrier_price_batch(self, r, KT_df, H):
        barrier_prices = KT_df.copy()

        r_interp = interp1d(r['T'], r['r'], kind='cubic', fill_value="extrapolate")
        barrier_prices['r'] = r_interp(barrier_prices['T'])

        knock = np.min(self.stockpaths, axis=0) <= H
        strikes_array = np.array(barrier_prices['K'])[:, np.newaxis]
        ST_array = self.stockpaths[np.round(barrier_prices['T']*self.n_steps,0).astype('int')]
        call_payoff = np.maximum(ST_array-strikes_array,0)
        put_payoff = np.maximum(strikes_array-ST_array,0)
        barrier_prices[f'DOBC_{H}'] = np.exp(-barrier_prices['T']*barrier_prices['r'])*np.mean(np.where(knock, 0, call_payoff), axis =1)
        barrier_prices[f'DIBC_{H}'] = np.exp(-barrier_prices['T']*barrier_prices['r'])*np.mean(np.where(~knock, 0, call_payoff), axis =1)
        barrier_prices[f'DOBP_{H}'] = np.exp(-barrier_prices['T']*barrier_prices['r'])*np.mean(np.where(knock, 0, put_payoff), axis =1)
        barrier_prices[f'DIBP_{H}'] = np.exp(-barrier_prices['T']*barrier_prices['r'])*np.mean(np.where(~knock, 0, put_payoff), axis =1)
        return barrier_prices
    
    def down_barrier_price_single(self, r, K, T, H):
        barrier_prices = {}
        knock = np.min(self.stockpaths, axis=0) <= H
        ST_array = self.stockpaths[np.round(T*self.n_steps,0).astype('int')]
        call_payoff = np.maximum(ST_array-K,0)
        put_payoff = np.maximum(K-ST_array,0)
        barrier_prices[f'DOBC_{H}'] = np.exp(-T*r)*np.mean(np.where(knock, 0, call_payoff))
        barrier_prices[f'DIBC_{H}'] = np.exp(-T*r)*np.mean(np.where(~knock, 0, call_payoff))
        barrier_prices[f'DOBP_{H}'] = np.exp(-T*r)*np.mean(np.where(knock, 0, put_payoff))
        barrier_prices[f'DIBP_{H}'] = np.exp(-T*r)*np.mean(np.where(~knock, 0, put_payoff))
        return barrier_prices