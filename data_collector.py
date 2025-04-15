"""
Data collection module for the Enhanced Managed Futures ETF Strategy.
Handles the collection of price data, economic indicators, and basic news data from AlphaVantage API.
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import os
import config
import utils

class DataCollector:
    """Class for collecting and processing data from AlphaVantage API."""
    
    def __init__(self, api_key=None, start_date=None, end_date=None):
        """
        Initialize the DataCollector.
        
        Parameters:
        -----------
        api_key : str, optional
            AlphaVantage API key (default: from config)
        start_date : str, optional
            Start date for data collection (default: from config)
        end_date : str, optional
            End date for data collection (default: from config)
        """
        self.api_key = api_key or config.API_KEY
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        
        if self.end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Define asset universe from config
        self.commodity_futures = config.COMMODITY_FUTURES
        self.currency_futures = config.CURRENCY_FUTURES
        self.bond_futures = config.BOND_FUTURES
        self.equity_futures = config.EQUITY_FUTURES
        
        # Combine all futures
        self.all_futures = {}
        self.all_futures.update(self.commodity_futures)
        self.all_futures.update(self.currency_futures)
        self.all_futures.update(self.bond_futures)
        self.all_futures.update(self.equity_futures)
        
        # Define asset classes
        self.asset_classes = {
            'Commodities': list(self.commodity_futures.keys()),
            'Currencies': list(self.currency_futures.keys()),
            'Bonds': list(self.bond_futures.keys()),
            'Equities': list(self.equity_futures.keys())
        }
        
        # Initialize data storage
        self.data = {
            'prices': None,
            'returns': None
        }
        self.economic_data = None
        self.news_data = None
        self.technical_indicators = {}
        
        # Create data directory
        utils.create_directory('data')
    
    def fetch_alphavantage_data(self, function, symbol='', interval='daily', outputsize='full', **kwargs):
        """
        Fetch data from AlphaVantage API with rate limiting.
        
        Parameters:
        -----------
        function : str
            API function to call
        symbol : str, optional
            Asset symbol (default: '')
        interval : str, optional
            Time interval (default: 'daily')
        outputsize : str, optional
            Output size (default: 'full')
        **kwargs : dict
            Additional parameters for the API call
        
        Returns:
        --------
        dict
            JSON response from the API
        """
        base_url = "https://www.alphavantage.co/query"
        
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the API request
        response = requests.get(base_url, params=params)
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                print(f"API Error for {function}, {symbol}: {data['Error Message']}")
                return None
            elif 'Information' in data and 'call frequency' in data['Information']:
                print("API call frequency exceeded. Waiting for 60 seconds.")
                time.sleep(60)  # Wait for a minute before retrying
                return self.fetch_alphavantage_data(function, symbol, interval, outputsize, **kwargs)
            
            return data
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return None
        
        # Sleep to respect API rate limits
        utils.rate_limit_sleep()
    
    def collect_commodity_data(self):
        """
        Collect commodity futures data.
        
        Returns:
        --------
        dict
            Dictionary of commodity price data frames
        """
        print("Collecting commodity data...")
        
        commodity_data = {}
        
        for symbol, name in self.commodity_futures.items():
            print(f"Fetching data for {name}...")
            
            # For WTI and Brent, use the commodities API
            if symbol in ['WTI', 'BRENT']:
                data = self.fetch_alphavantage_data(
                    function='WTI' if symbol == 'WTI' else 'BRENT',
                    symbol='',  # Not needed for commodities
                    interval='daily'
                )
                
                if data and 'data' in data:
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(data['data'])
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # Fix for invalid numeric values - replace '.' with NaN
                        df['value'] = df['value'].replace('.', np.nan)
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        
                        # Remove rows with NaN values
                        df = df.dropna(subset=['value'])
                        
                        # Rename column to match other data
                        df.rename(columns={'value': 'price'}, inplace=True)
                        
                        # Filter by date range
                        df = df[df.index >= self.start_date]
                        if self.end_date:
                            df = df[df.index <= self.end_date]
                        
                        if not df.empty:
                            commodity_data[symbol] = df
                        else:
                            print(f"Empty dataframe for {name} after filtering")
                    except Exception as e:
                        print(f"Error processing data for {name}: {str(e)}")
                        
                        # Fallback to generated data if processing fails
                        print(f"Using simulated data for {name} due to processing error")
                        # Generate simulated data (same as the else block below)
                        date_range = pd.date_range(start=self.start_date, end=self.end_date)
                        price = 100
                        prices = [price]
                        
                        for _ in range(1, len(date_range)):
                            change = np.random.normal(0, 0.015)
                            price *= (1 + change)
                            prices.append(price)
                        
                        df = pd.DataFrame({'price': prices}, index=date_range)
                        commodity_data[symbol] = df
                else:
                    print(f"Failed to fetch data for {name}")
                    # Continue with simulated data
                    print(f"Using simulated data for {name}")
                    date_range = pd.date_range(start=self.start_date, end=self.end_date)
                    price = 100
                    prices = [price]
                    
                    for _ in range(1, len(date_range)):
                        change = np.random.normal(0, 0.015)
                        price *= (1 + change)
                        prices.append(price)
                    
                    df = pd.DataFrame({'price': prices}, index=date_range)
                    commodity_data[symbol] = df
            
            # For COPPER
            elif symbol == 'COPPER':
                data = self.fetch_alphavantage_data(
                    function='COPPER',
                    symbol='',
                    interval='daily'
                )
                
                if data and 'data' in data:
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(data['data'])
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # Fix for invalid numeric values - replace '.' with NaN
                        df['value'] = df['value'].replace('.', np.nan)
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        
                        # Remove rows with NaN values
                        df = df.dropna(subset=['value'])
                        
                        # Rename column to match other data
                        df.rename(columns={'value': 'price'}, inplace=True)
                        
                        # Filter by date range
                        df = df[df.index >= self.start_date]
                        if self.end_date:
                            df = df[df.index <= self.end_date]
                        
                        if not df.empty:
                            commodity_data[symbol] = df
                        else:
                            print(f"Empty dataframe for {name} after filtering")
                    except Exception as e:
                        print(f"Error processing data for {name}: {str(e)}")
                        # Fallback to simulated data
                        print(f"Using simulated data for {name} due to processing error")
                        date_range = pd.date_range(start=self.start_date, end=self.end_date)
                        price = 100
                        prices = [price]
                        
                        for _ in range(1, len(date_range)):
                            change = np.random.normal(0, 0.015)
                            price *= (1 + change)
                            prices.append(price)
                        
                        df = pd.DataFrame({'price': prices}, index=date_range)
                        commodity_data[symbol] = df
                else:
                    print(f"Failed to fetch data for {name}")
                    # Continue with simulated data
                    print(f"Using simulated data for {name}")
                    date_range = pd.date_range(start=self.start_date, end=self.end_date)
                    price = 100
                    prices = [price]
                    
                    for _ in range(1, len(date_range)):
                        change = np.random.normal(0, 0.015)
                        price *= (1 + change)
                        prices.append(price)
                    
                    df = pd.DataFrame({'price': prices}, index=date_range)
                    commodity_data[symbol] = df
            
            # For other commodities, we'll use simulated data
            else:
                # Generate simulated data for other commodities
                date_range = pd.date_range(start=self.start_date, end=self.end_date)
                price = 100  # Starting price
                prices = [price]
                
                # Simple random walk with drift
                for _ in range(1, len(date_range)):
                    change = np.random.normal(0, 0.015)  # Daily volatility
                    
                    # Add commodity-specific characteristics
                    if symbol == 'NATURAL_GAS':
                        # More volatile
                        change *= 1.5
                    elif symbol in ['WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE']:
                        # Seasonal component
                        day_of_year = date_range[_].dayofyear
                        seasonal = 0.05 * np.sin(2 * np.pi * day_of_year / 365)
                        change += seasonal
                    
                    # Update price
                    price *= (1 + change)
                    prices.append(price)
                
                # Create DataFrame
                df = pd.DataFrame({'price': prices}, index=date_range)
                commodity_data[symbol] = df
        
        return commodity_data
    
    def collect_currency_data(self):
        """
        Collect currency futures data from AlphaVantage.
        
        Returns:
        --------
        dict
            Dictionary of currency price data frames
        """
        print("Collecting currency data...")
        
        currency_data = {}
        
        for symbol, name in self.currency_futures.items():
            print(f"Fetching data for {name}...")
            
            # Determine the from_currency and to_currency
            if symbol in ['EUR', 'GBP', 'AUD']:
                from_currency = symbol
                to_currency = 'USD'
            else:  # JPY, CAD
                from_currency = 'USD'
                to_currency = symbol
            
            # Fetch FX data - note the correct parameter structure for FX_DAILY
            data = self.fetch_alphavantage_data(
                function='FX_DAILY',
                symbol='',  # Not used for FX
                from_symbol=from_currency,
                to_symbol=to_currency,
                outputsize='full'
            )
            
            if data and 'Time Series FX (Daily)' in data:
                try:
                    # Convert to DataFrame
                    time_series = data['Time Series FX (Daily)']
                    df = pd.DataFrame(time_series).T
                    df.index = pd.DatetimeIndex(df.index)
                    
                    # Extract close price
                    df.columns = [col.split('. ')[1] for col in df.columns]
                    df['price'] = pd.to_numeric(df['close'], errors='coerce')
                    
                    # Filter by date range
                    df = df[df.index >= self.start_date]
                    if self.end_date:
                        df = df[df.index <= self.end_date]
                    
                    # Remove NaN values
                    df = df.dropna(subset=['price'])
                    
                    # For USD/XXX pairs, invert the price to get XXX/USD
                    if from_currency == 'USD':
                        df['price'] = 1 / df['price']
                    
                    if not df.empty:
                        currency_data[symbol] = df[['price']]
                    else:
                        print(f"Empty dataframe for {name} after filtering")
                        self._generate_simulated_currency_data(symbol, name, currency_data)
                except Exception as e:
                    print(f"Error processing data for {name}: {str(e)}")
                    self._generate_simulated_currency_data(symbol, name, currency_data)
            else:
                print(f"Failed to fetch data for {name}")
                self._generate_simulated_currency_data(symbol, name, currency_data)
        
        return currency_data

    def _generate_simulated_currency_data(self, symbol, name, currency_data):
        """Helper method to generate simulated currency data"""
        print(f"Using simulated data for {name}")
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        price = 1.0  # Starting price (exchange rate)
        prices = [price]
        
        # Simple random walk with drift
        for _ in range(1, len(date_range)):
            # Lower volatility for currencies
            change = np.random.normal(0, 0.005)
            price *= (1 + change)
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({'price': prices}, index=date_range)
        currency_data[symbol] = df
    
    def collect_bond_data(self):
        """
        Collect bond futures data (Treasury yields) from AlphaVantage.
        
        Returns:
        --------
        dict
            Dictionary of bond price data frames
        """
        print("Collecting bond data...")
        
        bond_data = {}
        
        # Fetch Treasury Yield data
        data = self.fetch_alphavantage_data(
            function='TREASURY_YIELD',
            symbol='',  # Not used for Treasury
            interval='daily',
            maturity='10year'  # 10-year Treasury as a proxy for bond futures
        )
        
        if data and 'data' in data:
            try:
                # Convert to DataFrame
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Fix for invalid numeric values - replace '.' with NaN
                df['value'] = df['value'].replace('.', np.nan)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Remove rows with NaN values
                df = df.dropna(subset=['value'])
                
                # Rename column to match other data
                df.rename(columns={'value': 'price'}, inplace=True)
                
                # Filter by date range
                df = df[df.index >= self.start_date]
                if self.end_date:
                    df = df[df.index <= self.end_date]
                
                # Convert yield to price (inverse relationship)
                # This is a simplification - in reality, the relationship is more complex
                df['price'] = 100 - df['price']  # Simple approximation
                
                if not df.empty:
                    bond_data['TREASURY_YIELD'] = df
                else:
                    print("Empty dataframe for Treasury Yield after filtering")
                    self._generate_simulated_bond_data(bond_data)
            except Exception as e:
                print(f"Error processing Treasury Yield data: {str(e)}")
                self._generate_simulated_bond_data(bond_data)
        else:
            print("Failed to fetch Treasury Yield data")
            self._generate_simulated_bond_data(bond_data)
        
        return bond_data

    def _generate_simulated_bond_data(self, bond_data):
        """Helper method to generate simulated bond data"""
        print("Using simulated data for Treasury Yield")
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        
        # For bond prices (not yields)
        price = 100.0  # Starting price for bond (par value)
        prices = [price]
        
        # Simple random walk with drift and mean reversion
        for _ in range(1, len(date_range)):
            # Less volatile for bonds
            change = np.random.normal(0, 0.002)
            
            # Add mean reversion to par
            mean_reversion = 0.05 * (100 - price) / 100
            
            price *= (1 + change + mean_reversion)
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({'price': prices}, index=date_range)
        bond_data['TREASURY_YIELD'] = df

    def collect_equity_data(self):
        """
        Collect equity futures data.
        
        Returns:
        --------
        dict
            Dictionary of equity price data frames
        """
        print("Collecting equity index data...")
        
        equity_data = {}
        
        for symbol, name in self.equity_futures.items():
            print(f"Fetching data for {name}...")
            
            # Fetch equity data
            data = self.fetch_alphavantage_data(
                function='TIME_SERIES_DAILY_ADJUSTED',
                symbol=symbol,
                outputsize='full'
            )
            
            if data and 'Time Series (Daily)' in data:
                # Convert to DataFrame
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame(time_series).T
                df.index = pd.DatetimeIndex(df.index)
                
                # Extract adjusted close price
                df.columns = [col.split('. ')[1] for col in df.columns]
                df['price'] = pd.to_numeric(df['adjusted close'])
                
                # Filter by date range
                df = df[df.index >= self.start_date]
                if self.end_date:
                    df = df[df.index <= self.end_date]
                
                equity_data[symbol] = df[['price']]
            else:
                print(f"Failed to fetch data for {name}")
                # Generate simulated data
                self._generate_simulated_equity_data(symbol, name, equity_data)
        
        return equity_data
    
    def _generate_simulated_equity_data(self, symbol, name, equity_data):
        """Helper method to generate simulated equity data"""
        print(f"Using simulated data for {name}")
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        price = 100  # Starting price
        prices = [price]
        
        # Simple random walk with upward drift
        for _ in range(1, len(date_range)):
            change = np.random.normal(0.0004, 0.01)  # Slight upward drift
            price *= (1 + change)
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({'price': prices}, index=date_range)
        equity_data[symbol] = df
    
    def collect_economic_data(self):
        """
        Collect economic indicator data from AlphaVantage with improved error handling.
        
        Returns:
        --------
        pd.DataFrame
            Economic data
        """
        print("Collecting economic data...")
        
        # Initialize empty DataFrame
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        economic_data = pd.DataFrame(index=dates)
        
        # Fetch Real GDP data
        real_gdp = self.fetch_alphavantage_data(
            function='REAL_GDP',
            symbol='',
            interval='quarterly'
        )
        
        if real_gdp and 'data' in real_gdp:
            try:
                # Process GDP data
                gdp_df = pd.DataFrame(real_gdp['data'])
                gdp_df['date'] = pd.to_datetime(gdp_df['date'])
                gdp_df.set_index('date', inplace=True)
                
                # Handle problematic values
                gdp_df['value'] = gdp_df['value'].replace('.', np.nan)
                gdp_df['value'] = pd.to_numeric(gdp_df['value'], errors='coerce')
                
                # Remove rows with NaN values
                gdp_df = gdp_df.dropna(subset=['value'])
                
                # Resample to daily frequency (forward fill)
                gdp_daily = gdp_df['value'].resample('D').ffill()
                economic_data['Real_GDP'] = gdp_daily
            except Exception as e:
                print(f"Error processing Real GDP data: {str(e)}")
        
        # Fetch Treasury Yield data (as an economic indicator, not as a tradable asset)
        treasury_yield = self.fetch_alphavantage_data(
            function='TREASURY_YIELD',
            symbol='',
            interval='daily',
            maturity='10year'
        )
        
        if treasury_yield and 'data' in treasury_yield:
            try:
                # Process Treasury Yield data
                yield_df = pd.DataFrame(treasury_yield['data'])
                yield_df['date'] = pd.to_datetime(yield_df['date'])
                yield_df.set_index('date', inplace=True)
                
                # Handle problematic values
                yield_df['value'] = yield_df['value'].replace('.', np.nan)
                yield_df['value'] = pd.to_numeric(yield_df['value'], errors='coerce')
                
                # Remove rows with NaN values
                yield_df = yield_df.dropna(subset=['value'])
                
                economic_data['Treasury_Yield'] = yield_df['value']
            except Exception as e:
                print(f"Error processing Treasury Yield data: {str(e)}")
        
        # Fetch CPI data
        cpi = self.fetch_alphavantage_data(
            function='CPI',
            symbol='',
            interval='monthly'
        )
        
        if cpi and 'data' in cpi:
            try:
                # Process CPI data
                cpi_df = pd.DataFrame(cpi['data'])
                cpi_df['date'] = pd.to_datetime(cpi_df['date'])
                cpi_df.set_index('date', inplace=True)
                
                # Handle problematic values
                cpi_df['value'] = cpi_df['value'].replace('.', np.nan)
                cpi_df['value'] = pd.to_numeric(cpi_df['value'], errors='coerce')
                
                # Remove rows with NaN values
                cpi_df = cpi_df.dropna(subset=['value'])
                
                # Resample to daily frequency (forward fill)
                cpi_daily = cpi_df['value'].resample('D').ffill()
                economic_data['CPI'] = cpi_daily
            except Exception as e:
                print(f"Error processing CPI data: {str(e)}")
        
        # Fetch Retail Sales data
        retail_sales = self.fetch_alphavantage_data(
            function='RETAIL_SALES',
            symbol='',
            interval='monthly'
        )
        
        if retail_sales and 'data' in retail_sales:
            try:
                # Process Retail Sales data
                retail_df = pd.DataFrame(retail_sales['data'])
                retail_df['date'] = pd.to_datetime(retail_df['date'])
                retail_df.set_index('date', inplace=True)
                
                # Handle problematic values
                retail_df['value'] = retail_df['value'].replace('.', np.nan)
                retail_df['value'] = pd.to_numeric(retail_df['value'], errors='coerce')
                
                # Remove rows with NaN values
                retail_df = retail_df.dropna(subset=['value'])
                
                # Resample to daily frequency (forward fill)
                retail_daily = retail_df['value'].resample('D').ffill()
                economic_data['Retail_Sales'] = retail_daily
            except Exception as e:
                print(f"Error processing Retail Sales data: {str(e)}")
        
        # Fetch Unemployment Rate data
        unemployment = self.fetch_alphavantage_data(
            function='UNEMPLOYMENT',
            symbol='',
            interval='monthly'
        )
        
        if unemployment and 'data' in unemployment:
            try:
                # Process Unemployment data
                unemp_df = pd.DataFrame(unemployment['data'])
                unemp_df['date'] = pd.to_datetime(unemp_df['date'])
                unemp_df.set_index('date', inplace=True)
                
                # Handle problematic values
                unemp_df['value'] = unemp_df['value'].replace('.', np.nan)
                unemp_df['value'] = pd.to_numeric(unemp_df['value'], errors='coerce')
                
                # Remove rows with NaN values
                unemp_df = unemp_df.dropna(subset=['value'])
                
                # Resample to daily frequency (forward fill)
                unemp_daily = unemp_df['value'].resample('D').ffill()
                economic_data['Unemployment'] = unemp_daily
            except Exception as e:
                print(f"Error processing Unemployment data: {str(e)}")
        
        # Fill any missing values
        economic_data = economic_data.ffill().bfill()
        
        # Create a simple Economic Policy Uncertainty Index based on volatility of indicators
        if len(economic_data.columns) > 0:
            try:
                # Calculate rolling standard deviation of available indicators
                std_cols = []
                for col in economic_data.columns:
                    std_col = f"{col}_std"
                    economic_data[std_col] = economic_data[col].rolling(window=30).std()
                    std_cols.append(std_col)
                
                # Average the standardized volatilities to create a simple EPU index
                if std_cols:
                    # Standardize each volatility measure
                    for col in std_cols:
                        if economic_data[col].std() > 0:
                            economic_data[col] = (economic_data[col] - economic_data[col].mean()) / economic_data[col].std()
                    
                    # Create EPU index as the average of standardized volatilities
                    economic_data['EPU_Index'] = economic_data[std_cols].mean(axis=1)
                    
                    # Scale to a more interpretable range (0-100)
                    min_val = economic_data['EPU_Index'].min()
                    max_val = economic_data['EPU_Index'].max()
                    if max_val > min_val:
                        economic_data['EPU_Index'] = 100 * (economic_data['EPU_Index'] - min_val) / (max_val - min_val)
                    else:
                        economic_data['EPU_Index'] = 50  # Default value if min=max
            except Exception as e:
                print(f"Error creating EPU Index: {str(e)}")
                economic_data['EPU_Index'] = 50  # Default value on error
        
        # Create a Tariff Impact Index based on recent policy changes
        try:
            # Initialize with a base level
            economic_data['Tariff_Impact'] = 50
            
            # Add increasing trend starting from 2018 (when US tariffs began to increase)
            tariff_start = max(pd.Timestamp('2018-01-01'), pd.Timestamp(self.start_date))
            days_since_start = (economic_data.index - tariff_start).days.astype(float)
            
            # Apply increasing tariff impact only to dates after tariff_start
            mask = economic_data.index >= tariff_start
            economic_data.loc[mask, 'Tariff_Impact'] = economic_data.loc[mask, 'Tariff_Impact'].astype(float) + days_since_start[mask] * 0.02
            
            # Add jumps at key tariff announcement dates
            for date, impact, _ in config.TARIFF_EVENTS:
                if pd.Timestamp(date) in economic_data.index:
                    economic_data.loc[date:, 'Tariff_Impact'] += impact
            
            # Smooth the tariff impact index
            economic_data['Tariff_Impact'] = economic_data['Tariff_Impact'].rolling(window=7).mean()
        except Exception as e:
            print(f"Error creating Tariff Impact index: {str(e)}")
            # Ensure tariff impact exists even on error
            if 'Tariff_Impact' not in economic_data.columns:
                economic_data['Tariff_Impact'] = 50
        
        # Clean and finalize
        economic_data = economic_data.ffill().bfill()
        
        # Keep only the main indicators for simplicity
        main_indicators = ['Real_GDP', 'Treasury_Yield', 'CPI', 'Unemployment', 'EPU_Index', 'Tariff_Impact']
        available_indicators = [col for col in main_indicators if col in economic_data.columns]
        
        if not available_indicators:
            print("No economic indicators available. Creating default indicators.")
            economic_data['EPU_Index'] = 50
            economic_data['Tariff_Impact'] = 50
            available_indicators = ['EPU_Index', 'Tariff_Impact']
        
        self.economic_data = economic_data[available_indicators]
        
        print("Economic data collection complete.")
        return self.economic_data
    
    def collect_news_data(self, keywords=None, days=30):
        """
        Collect news data (raw articles) for later sentiment analysis.
        This method only collects the news data and does not perform sentiment analysis.
        
        Parameters:
        -----------
        keywords : list, optional
            List of keywords to search for (default: tariff-related keywords)
        days : int, optional
            Number of days to look back (default: 30)
        
        Returns:
        --------
        pd.DataFrame
            Collected news data
        """
        print(f"Collecting news data for the past {days} days...")
        
        # Default keywords related to tariffs and trade
        if keywords is None:
            keywords = [
                'tariff', 'tariffs', 'trade war', 'trade dispute', 'trade tension', 
                'trade policy', 'protectionism', 'import duty', 'export control'
            ]
            
        # Convert keywords list to comma-separated string
        topics = ','.join(keywords).replace(' ', '_')
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch news from AlphaVantage
        data = self.fetch_alphavantage_data(
            function='NEWS_SENTIMENT',
            symbol='',
            topics=topics,
            time_from=start_date.strftime('%Y%m%dT%H%M'),
            time_to=end_date.strftime('%Y%m%dT%H%M'),
            sort='RELEVANCE'
        )
        
        if data and 'feed' in data:
            try:
                # Extract relevant information from articles
                articles = []
                for item in data['feed']:
                    # Extract basic info
                    article = {
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', ''),
                        'summary': item.get('summary', ''),
                        'overall_sentiment_score': item.get('overall_sentiment_score', 0),
                        'overall_sentiment_label': item.get('overall_sentiment_label', 'neutral')
                    }
                    
                    # Parse and format time
                    time_published = item.get('time_published', '')
                    if time_published:
                        try:
                            date = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                            article['date'] = date
                        except ValueError:
                            article['date'] = None
                    
                    # Extract topics
                    if 'topics' in item:
                        topics = [topic.get('topic', '') for topic in item['topics']]
                        article['topics'] = ', '.join(topics)
                    
                    articles.append(article)
                
                # Create DataFrame
                news_df = pd.DataFrame(articles)
                
                # Set date as index if available
                if 'date' in news_df.columns and not news_df['date'].isna().all():
                    news_df.set_index('date', inplace=True)
                    news_df.sort_index(inplace=True)
                
                self.news_data = news_df
                print(f"Collected {len(news_df)} news articles")
                return news_df
                
            except Exception as e:
                print(f"Error processing news data: {str(e)}")
                # Fall back to generating dummy data
                news_df = self._generate_dummy_news_data(start_date, end_date, keywords)
                self.news_data = news_df
                return news_df
        else:
            print("Failed to fetch news data or no articles found with the specified keywords.")
            # Generate dummy data
            news_df = self._generate_dummy_news_data(start_date, end_date, keywords)
            self.news_data = news_df
            return news_df
    
    def _generate_dummy_news_data(self, start_date, end_date, keywords):
        """
        Generate dummy news data for testing or when API fails.
        
        Parameters:
        -----------
        start_date : datetime
            Start date for generated news
        end_date : datetime
            End date for generated news
        keywords : list
            Keywords used for search
        
        Returns:
        --------
        pd.DataFrame
            Dummy news data
        """
        print("Generating dummy news data for testing/demonstration")
        
        # Generate random dates within the range
        days_range = (end_date - start_date).days
        random_days = np.random.randint(0, days_range, size=40)
        dates = [start_date + timedelta(days=day) for day in sorted(random_days)]
        
        # Define headline patterns
        positive_patterns = [
            "Progress Made in {country} Trade Negotiations",
            "Markets Rally as {country} Tariff Tensions Ease",
            "New Trade Deal with {country} to Reduce Tariffs"
        ]
        
        negative_patterns = [
            "US Imposes New Tariffs on {country} Imports",
            "{country} Retaliates with Tariffs on US Goods",
            "Trade War Escalates as {country} Threatens New Tariffs"
        ]
        
        neutral_patterns = [
            "Analysis: Impact of {country} Tariffs on Global Economy",
            "Experts Weigh In on {country} Trade Dispute",
            "{sector} Companies Adjust to New Tariff Reality"
        ]
        
        countries = ["China", "EU", "Canada", "Mexico", "Japan"]
        sectors = ["Steel", "Technology", "Automotive", "Agriculture"]
        
        # Generate headlines
        articles = []
        for date in dates:
            pattern_type = np.random.choice(["positive", "negative", "neutral"])
            if pattern_type == "positive":
                pattern = np.random.choice(positive_patterns)
                sentiment_score = np.random.uniform(0.1, 0.9)
                sentiment_label = "positive"
            elif pattern_type == "negative":
                pattern = np.random.choice(negative_patterns)
                sentiment_score = np.random.uniform(-0.9, -0.1)
                sentiment_label = "negative"
            else:
                pattern = np.random.choice(neutral_patterns)
                sentiment_score = np.random.uniform(-0.1, 0.1)
                sentiment_label = "neutral"
            
            country = np.random.choice(countries)
            sector = np.random.choice(sectors)
            
            headline = pattern.format(country=country, sector=sector)
            summary = f"This is a dummy summary about {country} and {sector} related to tariffs and trade tensions."
            
            article = {
                'date': date,
                'title': headline,
                'source': np.random.choice(["Reuters", "Bloomberg", "CNBC"]),
                'url': f"https://example.com/news/{date.strftime('%Y%m%d')}/{np.random.randint(1000, 9999)}",
                'summary': summary,
                'overall_sentiment_score': sentiment_score,
                'overall_sentiment_label': sentiment_label,
                'topics': 'tariffs, trade'
            }
            
            articles.append(article)
        
        # Create DataFrame
        news_df = pd.DataFrame(articles)
        news_df.set_index('date', inplace=True)
        news_df.sort_index(inplace=True)
        
        return news_df
    
    def calculate_technical_indicators(self):
        """
        Calculate technical indicators for assets.
        
        Returns:
        --------
        dict
            Dictionary of technical indicators
        """
        print("Calculating technical indicators...")
        
        # Get price data
        prices = self.data['prices']
        
        # Store technical indicators for each asset
        self.technical_indicators = {}
        
        for symbol in prices.columns:
            print(f"Calculating technical indicators for {symbol}...")
            
            indicators = {}
            
            for indicator in config.TECHNICAL_INDICATORS:
                # Skip BBANDS and OBV for certain assets that might not work well with them
                if indicator in ['BBANDS', 'OBV'] and symbol in self.bond_futures:
                    continue
                
                # Use AlphaVantage API to get technical indicators for equities
                # For other assets, calculate directly in Python
                
                if symbol in self.equity_futures:
                    # For equities, we can use the API directly
                    data = self.fetch_alphavantage_data(
                        function=indicator,
                        symbol=symbol,
                        interval='daily',
                        time_period=20,  # Default period
                        series_type='close'
                    )
                    
                    if data:
                        # Extract the indicator data
                        indicator_key = f"Technical Analysis: {indicator}"
                        if indicator_key in data:
                            # Convert to DataFrame
                            ind_df = pd.DataFrame(data[indicator_key]).T
                            ind_df.index = pd.DatetimeIndex(ind_df.index)
                            
                            # Convert columns to numeric
                            for col in ind_df.columns:
                                ind_df[col] = pd.to_numeric(ind_df[col])
                            
                            # Store the indicator
                            indicators[indicator] = ind_df
                else:
                    # For other assets, calculate indicators directly
                    price_series = prices[symbol].dropna()
                    
                    if indicator == 'SMA':
                        # Simple Moving Average
                        sma = price_series.rolling(window=20).mean()
                        indicators['SMA'] = pd.DataFrame({'SMA': sma})
                    
                    elif indicator == 'EMA':
                        # Exponential Moving Average
                        ema = price_series.ewm(span=20, adjust=False).mean()
                        indicators['EMA'] = pd.DataFrame({'EMA': ema})
                    
                    elif indicator == 'RSI':
                        # Relative Strength Index
                        delta = price_series.diff()
                        gain = delta.clip(lower=0)
                        loss = -delta.clip(upper=0)
                        
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        indicators['RSI'] = pd.DataFrame({'RSI': rsi})
                    
                    elif indicator == 'STOCH':
                        # Stochastic Oscillator
                        # Requires high, low, close data which we might not have
                        # For simplicity, we'll simulate it based on close prices
                        high_14d = price_series.rolling(window=14).max()
                        low_14d = price_series.rolling(window=14).min()
                        
                        # %K calculation
                        k = 100 * ((price_series - low_14d) / (high_14d - low_14d))
                        
                        # %D calculation (3-day SMA of %K)
                        d = k.rolling(window=3).mean()
                        
                        indicators['STOCH'] = pd.DataFrame({'SlowK': k, 'SlowD': d})
                    
                    elif indicator == 'ADX':
                        # Average Directional Index
                        # Simplified calculation since we don't have high/low data
                        # Just a placeholder for demonstration
                        adx = price_series.rolling(window=14).std() / price_series * 100
                        adx = adx.rolling(window=14).mean()
                        
                        indicators['ADX'] = pd.DataFrame({'ADX': adx})
            
            # Store indicators for this asset
            self.technical_indicators[symbol] = indicators
        
        print("Technical indicators calculation complete.")
        return self.technical_indicators
    
    def collect_all_data(self):
        """
        Collect all data needed for the strategy.
        
        Returns:
        --------
        dict
            Dictionary with all collected data
        """
        print("Collecting all data for the strategy...")
        
        # Collect futures data for each asset class
        commodity_data = self.collect_commodity_data()
        currency_data = self.collect_currency_data()
        bond_data = self.collect_bond_data()
        equity_data = self.collect_equity_data()
        
        # Combine all futures data
        prices = {}
        for data_dict in [commodity_data, currency_data, bond_data, equity_data]:
            for symbol, df in data_dict.items():
                if not df.empty:
                    prices[symbol] = df['price']
        
        # Convert to DataFrame
        price_df = pd.DataFrame(prices)
        
        # Calculate returns
        returns_df = price_df.pct_change(fill_method=None).dropna()
        
        # Store data
        self.data = {
            'prices': price_df,
            'returns': returns_df
        }
        
        # Collect economic data
        self.collect_economic_data()
        
        # Collect news data (raw, without sentiment analysis)
        self.collect_news_data()
        
        # Calculate technical indicators
        self.calculate_technical_indicators()
        
        # Save data to files
        self.save_data()
        
        print("Data collection complete.")
        return {
            'price_data': self.data,
            'economic_data': self.economic_data,
            'news_data': self.news_data,
            'technical_indicators': self.technical_indicators
        }
    
    def save_data(self):
        """
        Save collected data to CSV files.
        
        Returns:
        --------
        None
        """
        print("Saving data to files...")
        
        # Create data directory if it doesn't exist
        utils.create_directory('data')
        
        # Save price data
        if self.data['prices'] is not None:
            self.data['prices'].to_csv('data/prices.csv')
        
        # Save returns data
        if self.data['returns'] is not None:
            self.data['returns'].to_csv('data/returns.csv')
        
        # Save economic data
        if self.economic_data is not None:
            self.economic_data.to_csv('data/economic_data.csv')
        
        # Save news data
        if self.news_data is not None:
            self.news_data.to_csv('data/news_data.csv')
        
        # Save technical indicators
        if self.technical_indicators:
            for symbol, indicators in self.technical_indicators.items():
                for indicator_name, indicator_df in indicators.items():
                    # Create directory for each asset
                    utils.create_directory(f'data/indicators/{symbol}')
                    # Save indicator
                    indicator_df.to_csv(f'data/indicators/{symbol}/{indicator_name}.csv')
        
        print("Data saved to files.")
    
    def load_data(self):
        """
        Load data from CSV files.
        
        Returns:
        --------
        dict
            Dictionary with all loaded data
        """
        print("Loading data from files...")
        
        try:
            # Load price data
            self.data['prices'] = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
            
            # Load returns data
            self.data['returns'] = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)
            
            # Load economic data
            self.economic_data = pd.read_csv('data/economic_data.csv', index_col=0, parse_dates=True)
            
            # Load news data
            if os.path.exists('data/news_data.csv'):
                self.news_data = pd.read_csv('data/news_data.csv', index_col=0, parse_dates=True)
            
            # Load technical indicators
            self.technical_indicators = {}
            for symbol in self.all_futures.keys():
                indicators_dir = f'data/indicators/{symbol}'
                if os.path.exists(indicators_dir):
                    self.technical_indicators[symbol] = {}
                    for indicator_file in os.listdir(indicators_dir):
                        indicator_name = indicator_file.split('.')[0]
                        self.technical_indicators[symbol][indicator_name] = pd.read_csv(
                            f'{indicators_dir}/{indicator_file}',
                            index_col=0,
                            parse_dates=True
                        )
            
            print("Data loaded from files.")
            return {
                'price_data': self.data,
                'economic_data': self.economic_data,
                'news_data': self.news_data,
                'technical_indicators': self.technical_indicators
            }
        
        except FileNotFoundError:
            print("Data files not found. Please collect data first.")
            return None