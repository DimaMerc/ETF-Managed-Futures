# Enhanced Managed Futures ETF Strategy

A Python implementation of a managed futures ETF strategy specifically designed to navigate economic uncertainty, with particular focus on tariff-related market impacts.

## Overview

This project implements a systematic trading strategy for futures contracts across multiple asset classes using AI/ML techniques to adapt to changing market conditions. The strategy is specifically designed to perform well during periods of economic uncertainty, particularly those caused by tariff implementations and trade policy changes.

Key features:
- Multi-asset class approach (commodities, currencies, bonds, equities)
- Market regime detection using Gaussian Mixture Models
- Trend prediction with LSTM neural networks
- Technical indicator integration
- Economic uncertainty and tariff impact modeling
- News sentiment analysis
- Dynamic portfolio allocation
- Comprehensive backtesting and performance analysis

## Strategy Components

1. **Data Collection** (`data_collector.py`)
   - Fetches historical price data from AlphaVantage API
   - Collects economic indicators and builds uncertainty metrics
   - Analyzes news sentiment related to tariffs and trade policy

2. **Regime Detection** (`regime_detector.py`)
   - Identifies distinct market regimes using clustering techniques
   - Classifies regimes based on asset behavior and economic conditions
   - Specially designed to detect trade tension regimes

3. **Trend Prediction** (`trend_predictor.py`)
   - Implements LSTM networks for time series forecasting
   - Integrates technical indicators
   - Adjusts predictions based on economic uncertainty and news sentiment

4. **Portfolio Allocation** (`portfolio_allocator.py`)
   - Dynamic asset allocation based on market regime
   - Position sizing using risk measures
   - Configurable leverage and constraints

5. **Backtesting** (`backtest_engine.py`)
   - Simulates the strategy on historical data
   - Calculates performance metrics
   - Visualizes results

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- AlphaVantage API key

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/managed-futures-etf.git
   cd managed-futures-etf
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Configure API key
   Edit `config.py` to add your AlphaVantage API key:
   ```python
   API_KEY = "YOUR_ALPHAVANTAGE_API_KEY"
   ```

## Usage

The strategy can be run in different modes using the main script:

```
python main.py --mode [collect|train|predict|allocate|backtest|run_all]
```

### Modes:

- **collect**: Collect data from AlphaVantage API
- **train**: Train the regime detection and trend prediction models
- **predict**: Generate predictions using trained models
- **allocate**: Allocate portfolio based on predictions
- **backtest**: Run a historical backtest of the strategy
- **run_all**: Run the complete pipeline

### Additional parameters:

- `--start_date`: Start date for data collection (YYYY-MM-DD)
- `--end_date`: End date for data collection (YYYY-MM-DD)
- `--api_key`: AlphaVantage API key (override config)
- `--use_saved_data`: Use previously saved data instead of collecting new data
- `--use_saved_models`: Use previously trained models instead of training new ones

## Strategy Configuration

Edit `config.py` to adjust strategy parameters:

- Asset universe (which futures to trade)
- Risk management parameters (max position size, leverage)
- Model parameters (LSTM sequence length, regime clusters)
- Rebalance frequency
- Tariff event dates and impact

## Performance Analysis

The strategy is designed to:

- Provide crisis alpha during periods of market stress
- Adapt to changing trade policies and tariff implementations
- Identify and capitalize on economic regime shifts
- Maintain a diversified portfolio across asset classes
- Generate positive returns over full market cycles

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a research implementation and should not be used for actual trading without thorough validation and risk management. The strategy involves derivatives and leverage which can amplify both gains and losses.

## Acknowledgements

The strategy design is based on academic research on managed futures, ML applications in finance, and economic analysis of tariff impacts on financial markets.