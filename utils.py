"""
Utility functions for the Enhanced Managed Futures ETF Strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import time
import os

def ensure_date_format(date_str):
    """
    Ensure date string is in 'YYYY-MM-DD' format
    
    Parameters:
    -----------
    date_str : str
        Date string
        
    Returns:
    --------
    str
        Formatted date string
    """
    if date_str is None:
        return datetime.now().strftime('%Y-%m-%d')
    
    # Check if date is already in the right format
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        # Try to convert from other common formats
        for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If all fails, raise exception
        raise ValueError(f"Date format not recognized for {date_str}")

def date_range(start_date, end_date=None):
    """
    Generate a range of business days between start_date and end_date
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (default: today)
        
    Returns:
    --------
    pd.DatetimeIndex
        Range of business days
    """
    start_date = ensure_date_format(start_date)
    end_date = ensure_date_format(end_date) if end_date else datetime.now().strftime('%Y-%m-%d')
    
    return pd.date_range(start=start_date, end=end_date, freq='B')

def calculate_returns(prices):
    """
    Calculate returns from prices
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
        
    Returns:
    --------
    pd.DataFrame
        Returns data
    """
    return prices.pct_change().dropna()

def calculate_rolling_volatility(returns, window=20):
    """
    Calculate rolling volatility
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    window : int, optional
        Rolling window size (default: 20)
        
    Returns:
    --------
    pd.DataFrame
        Rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_correlation_matrix(returns, window=60):
    """
    Calculate rolling correlation matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    window : int, optional
        Rolling window size (default: 60)
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return returns.rolling(window=window).corr()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    risk_free_rate : float, optional
        Annualized risk-free rate (default: 0.0)
        
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    excess_return = returns.mean() - risk_free_rate / 252
    return excess_return / returns.std() * np.sqrt(252)

def calculate_drawdown(returns):
    """
    Calculate drawdown
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
        
    Returns:
    --------
    pd.Series
        Drawdown series
    """
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown

def calculate_calmar_ratio(returns, period=252):
    """
    Calculate Calmar ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    period : int, optional
        Number of days to annualize (default: 252)
        
    Returns:
    --------
    float
        Calmar ratio
    """
    drawdown = calculate_drawdown(returns)
    max_drawdown = drawdown.min()
    if max_drawdown == 0:
        return np.inf
    
    return (returns.mean() * period) / abs(max_drawdown)

def prepare_lstm_data(series, seq_length):
    """
    Prepare data for LSTM model
    
    Parameters:
    -----------
    series : np.array
        Time series data
    seq_length : int
        Sequence length
        
    Returns:
    --------
    tuple
        (X, y) where X is the sequence data and y is the target
    """
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

def create_directory(directory):
    """
    Create a directory if it doesn't exist
    
    Parameters:
    -----------
    directory : str
        Directory path
        
    Returns:
    --------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def rate_limit_sleep(n_seconds=12):
    """
    Sleep to respect API rate limits
    
    Parameters:
    -----------
    n_seconds : int, optional
        Number of seconds to sleep (default: 12)
        
    Returns:
    --------
    None
    """
    time.sleep(n_seconds)

def generate_synthetic_data(start_date='2018-01-01', end_date=None, instruments=None, seed=42):
    """
    Generate synthetic price and return data for backtesting when real data is unavailable.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    print("Generating synthetic market data for backtesting...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Default end date is today
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate daily dates (business days only)
    dates = pd.date_range(start=start, end=end, freq='B')
    
    # Default instruments if none provided
    if instruments is None:
        instruments = {
            'Commodities': ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE'],
            'Currencies': ['EUR', 'GBP', 'JPY', 'CAD', 'AUD'],
            'Bonds': ['TREASURY_YIELD'],
            'Equities': ['SPY', 'QQQ', 'DIA', 'IWM']
        }
        # Flatten to a single list
        instruments = [item for sublist in instruments.values() for item in sublist]
    
    # Initialize price DataFrame
    prices_df = pd.DataFrame(index=dates)
    
    # Simple method to generate uncorrelated random walks
    # Using a more robust approach that doesn't rely on matrix operations
    for instrument in instruments:
        # Simple random walk with drift
        price = 100
        prices = [price]
        
        for i in range(1, len(dates)):
            # Different characteristics for different asset classes
            if instrument in ['WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE']:
                # Commodities: more volatile
                change = np.random.normal(0.0002, 0.012)
            elif instrument in ['EUR', 'GBP', 'JPY', 'CAD', 'AUD']:
                # Currencies: less volatile
                change = np.random.normal(0.0001, 0.005)
            elif instrument in ['TREASURY_YIELD']:
                # Bonds: negative correlation with equities, lower volatility
                change = np.random.normal(-0.0001, 0.004)
            else:
                # Equities: market-like returns
                change = np.random.normal(0.0003, 0.010)
                
            # Add some autocorrelation (momentum effect)
            if i > 1:
                change += 0.1 * (prices[-1] / prices[-2] - 1)
                
            # Add some cyclicality
            cyclical = 0.002 * np.sin(2 * np.pi * i / 252)  # Annual cycle
            
            # Update price
            price *= (1 + change + cyclical)
            prices.append(price)
        
        prices_df[instrument] = prices
    
    # Add correlations between related instruments
    # This is a simplified approach without matrix operations
    related_pairs = [
        ('WTI', 'BRENT', 0.9),           # Oil types are highly correlated
        ('SPY', 'QQQ', 0.8),             # Equity indices correlate
        ('SPY', 'DIA', 0.85),
        ('QQQ', 'DIA', 0.75),
        ('EUR', 'GBP', 0.6),             # Currency pairs
        ('WHEAT', 'CORN', 0.7),          # Agricultural commodities
        ('COTTON', 'SUGAR', 0.5)
    ]
    
    # Apply correlations by blending related price series
    for asset1, asset2, corr in related_pairs:
        if asset1 in prices_df.columns and asset2 in prices_df.columns:
            # Simple way to induce correlation - blend returns
            returns1 = prices_df[asset1].pct_change().fillna(0)
            returns2 = prices_df[asset2].pct_change().fillna(0)
            
            # Blend returns to create correlation
            blended_returns2 = corr * returns1 + (1 - corr) * returns2
            
            # Reconstruct price series from blended returns
            new_prices = [prices_df[asset2].iloc[0]]
            for ret in blended_returns2[1:]:
                new_prices.append(new_prices[-1] * (1 + ret))
            
            prices_df[asset2] = new_prices
    
    # Calculate returns
    returns_df = prices_df.pct_change().dropna()
    
    print(f"Generated synthetic data for {len(instruments)} instruments from {start_date} to {end_date}")
    return prices_df, returns_df