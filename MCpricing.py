import numpy as np

def BSdynamics(n_steps, n_paths, S0, r, q, T, params):
    dt = 1/n_steps
    sigma = params['sigma']
    # Time grid
    t = np.linspace(0, T, n_steps + 1)  # [0, dt, 2dt, ..., T]

    total_steps = n_steps*T
    # Generate random Brownian increments (Z ~ N(0, 1))
    Z = np.random.standard_normal((total_steps, n_paths))

    # Compute cumulative returns
    drift = (r - q -0.5 * sigma**2) * dt
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
    Z = np.random.standard_normal((total_steps, n_paths))
    
    # BS part
    drift = (r - q -0.5 * sigma**2) * dt
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
    Z = np.random.standard_normal((total_steps, n_paths))
    # Generate correlated Brownian motions
    Z_v = rho * Z + np.sqrt(1 - rho**2) * np.random.standard_normal((total_steps, n_paths))

    # Simulate volatility paths
    v = np.zeros((n_steps + 1, n_paths))
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
    returns = np.exp((r - q - 0.5 * v) * dt + np.sqrt(v * dt) * Z)
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
    Z = np.random.standard_normal((total_steps, n_paths))
    # Generate correlated Brownian motions
    Z_v = rho * Z + np.sqrt(1 - rho**2) * np.random.standard_normal((total_steps, n_paths))

    # Simulate volatility paths
    v = np.zeros((n_steps + 1, n_paths))
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
    returns = np.exp((r - q - 0.5 * v[:-1]) * dt + np.sqrt(v[:-1] * dt) * Z) * (1 + jump_shocks * (jump_sizes - 1))

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

    def simulate(self,r,q, model=None, dismethod=None, params=None):
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

        return self.stockpaths, self.returns
