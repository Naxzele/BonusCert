import numpy as np
import pandas as pd

def cleanup(df):
    """
    Cleans 0s and NAs
    
    Parameters:
    - df: DataFrame with ['bid', 'ask']
    
    Returns:
    - DataFrame with selected options
    """
    df_new = df[(df['bid']!=0) & (df['ask']!=0) & (df['bid'].notna()) & (df['ask'].notna()) & (df['openInterest'].notna())].copy()
    return df_new

def add_T(df, t0):
    """
    Converts maturities to T in years
    
    Parameters:
    - df: DataFrame with ['expiration']
    - t0: Initial date in YYYY-MM-DD
    
    Returns:
    - DataFrame with selected options
    """
    t0 = pd.to_datetime(t0)
    df['T'] = (pd.to_datetime(df['expiration'])-t0).dt.days/365
    return print('Maturity converted to years')

def width_function(T, a=0.2, b=2.5, c=0.05):
    return a * np.exp(-b * T) + c

def build_calibration_triangle(df, S0, a=0.2, b=2.5, c=0.05):
    """
    Selects a triangular subset of option data using a width function.
    
    Parameters:
    - df: DataFrame with ['T', 'K']
    - S0: current spot price
    - width_func: function T -> width (e.g., 0.2 → ±20% of spot)
    
    Returns:
    - DataFrame with selected options
    """
    # df_new = df[['T','K']].copy()
    df['width'] = width_function(df['T'], a, b, c)
    df['upperK'] = (1 + df['width'])*S0
    df['lowerK'] = (1 - df['width'])*S0

    df_final = df[(df['K'] >= df['lowerK'] ) & (df['K'] <= df['upperK'])].copy()

    return df_final