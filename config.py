"""
Configuration parameters for the Enhanced Managed Futures ETF Strategy.
"""

# API Settings
API_KEY = "80HLC8UDC38Z6HVE"  

# Strategy Parameters
START_DATE = "2018-01-01"  # Start date for historical data
END_DATE = None  # End date for historical data (None means today)
LOOKBACK_PERIOD = 252  # Number of trading days to look back for training models
REBALANCE_FREQUENCY = 21  # Number of trading days between portfolio rebalances

# Model Parameters
LSTM_SEQUENCE_LENGTH = 60  # Sequence length for LSTM models
LSTM_EPOCHS = 50  # Number of epochs for training LSTM models
REGIME_CLUSTERS = 4  # Number of regime clusters to identify

# Risk Management Parameters
MAX_POSITION_SIZE = 0.20  # Maximum weight for any single position (long or short)
LEVERAGE = 1.0  # Total leverage (1.0 means no leverage)

# Asset Universe
COMMODITY_FUTURES = {
    'WTI': 'Crude Oil (WTI)',
    'BRENT': 'Crude Oil (Brent)', 
    'NATURAL_GAS': 'Natural Gas',
    'COPPER': 'Copper',
    'ALUMINUM': 'Aluminum',
    'WHEAT': 'Wheat',
    'CORN': 'Corn', 
    'COTTON': 'Cotton',
    'SUGAR': 'Sugar',
    'COFFEE': 'Coffee'
}

CURRENCY_FUTURES = {
    'EUR': 'Euro',               # EUR/USD
    'GBP': 'British Pound',      # GBP/USD
    'JPY': 'Japanese Yen',       # USD/JPY
    'CAD': 'Canadian Dollar',    # USD/CAD
    'AUD': 'Australian Dollar'   # AUD/USD
}

BOND_FUTURES = {
    'TREASURY_YIELD': 'Treasury Yield',  # 10-year Treasury Yield as proxy
}

EQUITY_FUTURES = {
    'SPY': 'S&P 500',            # S&P 500 ETF
    'QQQ': 'Nasdaq',             # Nasdaq ETF
    'DIA': 'Dow Jones',          # Dow Jones ETF
    'IWM': 'Russell 2000'        # Russell 2000 ETF
}

# Technical Indicators to use
TECHNICAL_INDICATORS = [
    'SMA',     # Simple Moving Average
    'EMA',     # Exponential Moving Average
    'RSI',     # Relative Strength Index
    'STOCH',   # Stochastic Oscillator
    'ADX',     # Average Directional Movement Index
    'CCI',     # Commodity Channel Index
    'AROON',   # Aroon Indicator
    'BBANDS',  # Bollinger Bands
    'OBV'      # On-Balance Volume
]

# Key tariff implementation dates for economic analysis
TARIFF_EVENTS = [
    ('2018-03-01', 10, 'Steel and Aluminum tariffs'),
    ('2018-07-06', 15, 'China tariffs begin'),
    ('2019-05-10', 20, 'China tariff increase'),
    ('2019-08-01', 25, 'Additional China tariffs'),
    ('2020-01-15', -15, 'Phase One trade deal (decrease)'),
    ('2023-05-11', 20, 'Proposed new vehicle tariffs'),
    ('2024-01-31', 25, 'Increased China tariffs'),
    ('2024-03-15', 15, 'Steel and aluminum tariff increase'),
]

# Backtesting parameters
INITIAL_CAPITAL = 1000000  # Initial capital for backtesting