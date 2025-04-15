"""
Main execution script for the Enhanced Managed Futures ETF Strategy.

This script ties together all the components of the strategy:
1. Data collection from AlphaVantage API
2. Market regime detection
3. Trend prediction
4. Portfolio allocation
5. Backtesting and evaluation

Usage:
    python main.py --mode [collect|train|predict|allocate|backtest|run_all]
    
    Modes:
    - collect: Collect data from AlphaVantage API
    - train: Train the regime detection and trend prediction models
    - predict: Generate predictions using trained models
    - allocate: Allocate portfolio based on predictions
    - backtest: Run a historical backtest of the strategy
    - run_all: Run the complete pipeline
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import strategy components
import config
import utils
from data_collector import DataCollector
from regime_detector import EnhancedRegimeDetector
from trend_predictor import TrendPredictor
from portfolio_allocator import BalancedPortfolioAllocator
from backtest_engine import BacktestEngine
from volatility_targeting import VolatilityTargeting
from tariff_sentiment_analyzer import TariffSentimentAnalyzer  # Make sure filename matches

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Managed Futures ETF Strategy')
    
    parser.add_argument('--mode', type=str, default='run_all',
                        choices=['collect', 'train', 'predict', 'allocate', 'backtest', 'run_all'],
                        help='Mode of operation')
    
    parser.add_argument('--start_date', type=str, default=config.START_DATE,
                        help='Start date for data collection (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=config.END_DATE,
                        help='End date for data collection (YYYY-MM-DD)')
    
    parser.add_argument('--api_key', type=str, default=config.API_KEY,
                        help='AlphaVantage API key')
    
    parser.add_argument('--use_saved_data', action='store_true',
                        help='Use previously saved data instead of collecting new data')
    
    parser.add_argument('--use_saved_models', action='store_true',
                        help='Use previously trained models instead of training new ones')
    
    return parser.parse_args()

def collect_data(args):
    """Collect data from AlphaVantage API."""
    print("\n" + "="*80)
    print("STEP 1: COLLECTING DATA")
    print("="*80)
    
    # Initialize data collector
    collector = DataCollector(
        api_key=args.api_key,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Collect or load data
    if args.use_saved_data:
        print("Loading data from saved files...")
        data = collector.load_data()
        if data is None:
            print("No saved data found. Collecting new data...")
            data = collector.collect_all_data()
    else:
        print("Collecting new data from AlphaVantage API...")
        data = collector.collect_all_data()
    
    return data

def detect_regimes(returns_data, economic_data=None):
    """Detect market regimes."""
    print("\n" + "="*80)
    print("STEP 2: DETECTING MARKET REGIMES")
    print("="*80)
    
    # Initialize regime detector
    regime_detector = EnhancedRegimeDetector(returns_data, economic_data)
    
    # Check if saved regimes exist and whether to use them
    if os.path.exists('data/regimes.csv'):
        print("Found saved regime file.")
        try:
            # Try to load regimes
            loaded_regimes, loaded_named_regimes = regime_detector.load_regimes()
            
            # Check if the loaded regimes align with our current data
            if loaded_regimes is not None and loaded_named_regimes is not None:
                # Check date ranges
                if loaded_regimes.index.equals(returns_data.index):
                    print("Using saved regimes - indices match.")
                else:
                    print("Saved regimes have different date range than current data.")
                    print(f"Saved regimes: {loaded_regimes.index[0]} to {loaded_regimes.index[-1]}")
                    print(f"Current data: {returns_data.index[0]} to {returns_data.index[-1]}")
                    
                    # Option 1: Generate new regimes
                    print("Detecting new regimes for current data...")
                    regime_detector.detect_regimes()
                    regime_detector.save_regimes()
        except Exception as e:
            print(f"Error loading saved regimes: {str(e)}")
            print("Detecting new regimes...")
            regime_detector.detect_regimes()
            regime_detector.save_regimes()
    else:
        print("No saved regimes found. Detecting market regimes...")
        regime_detector.detect_regimes()
        regime_detector.save_regimes()
    
    # Skip visualization if we're dealing with synthetic data
    # This avoids the visualization error but still keeps the regime data
    is_synthetic = hasattr(regime_detector, 'is_synthetic_data') and regime_detector.is_synthetic_data
    
    if not is_synthetic:
        try:
            regime_detector.visualize_regimes()
        except Exception as e:
            print(f"Warning: Could not visualize regimes: {str(e)}")
            print("This is expected with synthetic data - continuing without visualization.")
    
    return regime_detector

def predict_trends(returns_data, technical_indicators=None, use_saved_models=False):
    """Predict price trends."""
    print("\n" + "="*80)
    print("STEP 3: PREDICTING PRICE TRENDS")
    print("="*80)
    
    # Initialize trend predictor
    trend_predictor = TrendPredictor(returns_data, technical_indicators)
    
    # Build or load models
    if use_saved_models and os.path.exists('models'):
        print("Loading trend models from saved files...")
        trend_predictor.load_trend_models()
    else:
        print("Building trend prediction models...")
        trend_predictor.build_trend_models()
        trend_predictor.save_trend_models()
    
    # Generate predictions - no need to capture return values
    trend_predictor.predict_trends()
    
    return trend_predictor

def allocate_portfolio(trend_predictor, regime_detector, economic_data, news_data, returns_data):
    """Allocate portfolio based on predictions and regimes."""
    print("\n" + "="*80)
    print("STEP 4: ALLOCATING PORTFOLIO")
    print("="*80)
    
    # First check if we have predictions - if not, generate them
    if not hasattr(trend_predictor, 'trend_direction') or trend_predictor.trend_direction is None:
        print("No predictions found. Generating trend predictions...")
        trend_predictor.predict_trends()
    
    # Check if we have predictions now
    if not hasattr(trend_predictor, 'trend_direction') or trend_predictor.trend_direction is None:
        print("Failed to generate predictions. Using default values.")
        # Create default predictions (all assets with weak positive trend)
        # This is a fallback to prevent crashes
        if hasattr(trend_predictor, 'returns') and trend_predictor.returns is not None:
            assets = trend_predictor.returns.columns
            trend_predictor.trend_direction = pd.Series(1, index=assets)  # Default to positive trend
            trend_predictor.trend_strength = pd.Series(0.1, index=assets)  # Default to weak signal
            trend_predictor.final_direction = trend_predictor.trend_direction
            trend_predictor.final_strength = trend_predictor.trend_strength
        else:
            print("Error: No asset data available.")
            return None
    else:
        # Adjust trend predictions for economic uncertainty and news sentiment
        try:
            trend_predictor.adjust_for_economic_uncertainty(economic_data)
        except Exception as e:
            print(f"Error adjusting for economic uncertainty: {str(e)}")
            print("Using unadjusted predictions.")
            trend_predictor.adjusted_direction = trend_predictor.trend_direction
            trend_predictor.adjusted_strength = trend_predictor.trend_strength
        
        try:
            trend_predictor.adjust_for_news_sentiment(news_data)
        except Exception as e:
            print(f"Error adjusting for news sentiment: {str(e)}")
            print("Using economically adjusted predictions.")
            trend_predictor.final_direction = trend_predictor.adjusted_direction
            trend_predictor.final_strength = trend_predictor.adjusted_strength
    
    # Use final_direction and final_strength if they exist, otherwise fall back to trend_direction/strength
    direction = getattr(trend_predictor, 'final_direction', 
                      getattr(trend_predictor, 'adjusted_direction',
                             trend_predictor.trend_direction))
    
    strength = getattr(trend_predictor, 'final_strength',
                     getattr(trend_predictor, 'adjusted_strength',
                            trend_predictor.trend_strength))
    
    # Get current regime
    current_regime = regime_detector.get_current_regime()
    print(f"Current market regime: {current_regime}")
    
    # Allocate portfolio
    portfolio_allocator = BalancedPortfolioAllocator(
        direction,
        strength,
        regime=current_regime
    )
    
    # Get initial weights
    weights = portfolio_allocator.allocate_portfolio()
    
    # Adjust weights based on volatility targeting
    adjusted_weights = adjust_for_volatility(
        returns_data=returns_data,  
        weights=weights,
        regime=current_regime,
        target_volatility=0.10
    )
    
    # Update the allocator's weights with volatility-adjusted weights
    portfolio_allocator.weights = adjusted_weights
    
    # Visualize allocation
    portfolio_allocator.visualize_allocation()
    
    # Save allocation
    portfolio_allocator.save_allocation()
    
    return portfolio_allocator

def adjust_for_volatility(returns_data, weights, regime=None, target_volatility=0.10):
    """
    Adjust portfolio weights based on volatility targeting.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Historical returns data
    weights : pd.Series
        Current portfolio weights
    regime : str, optional
        Current market regime (default: None)
    target_volatility : float, optional
        Target annualized volatility (default: 0.10)
        
    Returns:
    --------
    pd.Series
        Volatility-adjusted weights
    """
    print("\n" + "="*80)
    print("STEP 4.5: APPLYING VOLATILITY TARGETING")
    print("="*80)
    
    # Initialize volatility targeter
    vol_targeter = VolatilityTargeting(
        returns_data=returns_data,
        target_volatility=target_volatility,
        max_leverage=config.LEVERAGE * 1.5,  # Allow some extra leverage
        min_leverage=0.5,  # Minimum leverage
        current_regime=regime
    )
    
    # Calculate historical volatility
    vol_targeter.calculate_historical_volatility()
    
    # Predict future volatility
    vol_targeter.predict_volatility()
    
    # Calculate target leverage
    target_leverage = vol_targeter.calculate_target_leverage()
    
    # Adjust weights
    adjusted_weights = vol_targeter.adjust_portfolio_weights(weights)
    
    # Visualize volatility targeting
    vol_targeter.visualize_volatility_targeting()
    
    return adjusted_weights

def run_backtest(price_data, returns_data, regimes=None, economic_data=None):
    """Run a historical backtest of the strategy."""
    print("\n" + "="*80)
    print("STEP 5: BACKTESTING THE STRATEGY")
    print("="*80)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        price_data,
        returns_data,
        regimes=regimes,
        economic_data=economic_data
    )
    
    # Run backtest
    results = backtest_engine.backtest_strategy()
    
    # Only visualize and save results if backtest was successful
    if results is not None:
        # Visualize results
        backtest_engine.visualize_results()
        
        # Save results
        backtest_engine.save_results()
    else:
        print("Backtest failed or no returns data available. Skipping visualization and saving.")
    
    return backtest_engine

def print_summary(portfolio_allocator, backtest_engine):
    """Print a summary of the strategy with robust error handling for missing data."""
    print("\n" + "="*80)
    print("STRATEGY SUMMARY")
    print("="*80)
    
    # Check if portfolio_allocator has valid weights
    if portfolio_allocator is None or not hasattr(portfolio_allocator, 'weights') or portfolio_allocator.weights is None:
        print("\nNo valid portfolio allocation available.")
    else:
        # Current allocation
        print("\nCurrent Portfolio Allocation:")
        weights = portfolio_allocator.weights
        non_zero_weights = weights[weights != 0].sort_values(ascending=False)
        
        print(f"Number of positions: {len(non_zero_weights)}")
        
        # Check if get_long_short_exposure method exists and works
        try:
            long_exp, short_exp = portfolio_allocator.get_long_short_exposure()
            print(f"Long exposure: {long_exp:.2%}")
            print(f"Short exposure: {short_exp:.2%}")
        except Exception as e:
            print("Could not calculate long/short exposure")
        
        # Print top positions if available
        try:
            print("\nTop 5 Long Positions:")
            top_long, top_short = portfolio_allocator.get_top_positions(5)
            
            if len(top_long) > 0:
                for ticker, weight in top_long.items():
                    print(f"  {ticker}: {weight:.2%}")
            else:
                print("  No long positions")
            
            print("\nTop 5 Short Positions:")
            if len(top_short) > 0:
                for ticker, weight in top_short.items():
                    print(f"  {ticker}: {weight:.2%}")
            else:
                print("  No short positions")
        except Exception as e:
            print("Could not retrieve top positions")
    
    # Backtest performance
    print("\nBacktest Performance:")
    
    # Check if backtest_engine has valid performance metrics
    if backtest_engine is None or not hasattr(backtest_engine, 'performance') or backtest_engine.performance is None:
        print("No valid backtest performance data available.")
    else:
        try:
            performance = backtest_engine.performance
            
            # Check if required metrics exist
            if 'total_return' in performance:
                print(f"Total Return: {performance['total_return']:.2%}")
            
            if 'annualized_return' in performance:
                print(f"Annualized Return: {performance['annualized_return']:.2%}")
            
            if 'annualized_volatility' in performance:
                print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
            
            if 'sharpe_ratio' in performance:
                print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            
            if 'sortino_ratio' in performance:
                print(f"Sortino Ratio: {performance['sortino_ratio']:.2f}")
            
            if 'max_drawdown' in performance:
                print(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
            
            if 'win_rate' in performance:
                print(f"Win Rate (Monthly): {performance['win_rate']:.2%}")
            
        except Exception as e:
            print(f"Error displaying performance metrics: {str(e)}")
    
    print("\nNote: The strategy is designed to perform well in environments with high economic uncertainty,")
    print("particularly related to tariffs. Current allocation reflects the present market regime")
    if hasattr(portfolio_allocator, 'regime') and portfolio_allocator.regime is not None:
        print(f"({portfolio_allocator.regime}) and economic conditions.")
    else:
        print("and economic conditions.")


def main():
    """Main function to run the strategy."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up directory structure
    utils.create_directory('data')
    utils.create_directory('models')
    utils.create_directory('results')
    
    # Display start banner
    print("\n" + "*"*80)
    print("*" + " "*30 + "MANAGED FUTURES ETF STRATEGY" + " "*30 + "*")
    print("*" + " "*30 + "Optimized for Tariff Uncertainty" + " "*28 + "*")
    print("*"*80)
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date if args.end_date else 'Today'}")
    print(f"Mode: {args.mode}")
    print("*"*80 + "\n")
    
    # Execute the specified mode
    if args.mode in ['collect', 'run_all']:
        # Step 1: Collect data
        data = collect_data(args)
        price_data = data['price_data']['prices']
        returns_data = data['price_data']['returns']
        economic_data = data.get('economic_data', None)
        news_data = data.get('news_data', None)  # Changed from news_sentiment
        technical_indicators = data.get('technical_indicators', None)
    else:
        # Load saved data
        collector = DataCollector()
        data = collector.load_data()
        if data is None:
            print("No saved data found. Please run with --mode collect first.")
            sys.exit(1)
        
        price_data = data['price_data']['prices']
        returns_data = data['price_data']['returns']
        economic_data = data.get('economic_data', None)
        news_data = data.get('news_data', None)  # Changed from news_sentiment
        technical_indicators = data.get('technical_indicators', None)
    
    # Step 1.5: Check if we have valid data and generate synthetic data if needed
    if returns_data is None or returns_data.empty or len(returns_data.columns) < 2:
        print("Warning: Insufficient real market data. Generating synthetic data for backtesting.")
        # Get instrument lists from config
        instruments = list(config.COMMODITY_FUTURES.keys()) + list(config.CURRENCY_FUTURES.keys()) + \
                     list(config.BOND_FUTURES.keys()) + list(config.EQUITY_FUTURES.keys())
        # Generate synthetic data
        price_data, returns_data = utils.generate_synthetic_data(
            start_date=args.start_date,
            end_date=args.end_date,
            instruments=instruments
        )
        print(f"Using synthetic data with {len(returns_data)} days and {len(returns_data.columns)} instruments")
        
        # Also generate synthetic economic data if needed
        if economic_data is None or economic_data.empty:
            print("Generating synthetic economic data...")
            # Create simple synthetic economic data
            dates = returns_data.index
            economic_data = pd.DataFrame(index=dates)
            # Add EPU Index (Economic Policy Uncertainty)
            np.random.seed(42)
            base_uncertainty = np.cumsum(np.random.normal(0, 0.05, len(dates)))
            economic_data['EPU_Index'] = 50 + 20 * base_uncertainty / np.max(abs(base_uncertainty))
            # Add Tariff Impact
            tariff_impact = 50 * np.ones(len(dates))
            # Add jumps at certain dates
            tariff_dates = [
                pd.Timestamp('2018-03-01'), pd.Timestamp('2018-07-06'),
                pd.Timestamp('2019-05-10'), pd.Timestamp('2019-08-01')
            ]
            for date in dates:
                if date in tariff_dates:
                    idx = dates.get_loc(date)
                    tariff_impact[idx:] += 10  # Step increase at each tariff date
            economic_data['Tariff_Impact'] = tariff_impact
    
    if args.mode in ['train', 'run_all']:
        # Step 2: Detect market regimes
        regime_detector = detect_regimes(returns_data, economic_data)
        
        # Step 3: Predict price trends
        trend_predictor = predict_trends(returns_data, technical_indicators, args.use_saved_models)
    else:
        # Load saved models
        regime_detector = EnhancedRegimeDetector(returns_data, economic_data)
        regime_detector.load_regimes()
        
        trend_predictor = TrendPredictor(returns_data, technical_indicators)
        trend_predictor.load_trend_models()
        trend_predictor.predict_trends()
    
    if args.mode in ['predict', 'allocate', 'run_all']:
        # Step 4: Allocate portfolio
        portfolio_allocator = allocate_portfolio(trend_predictor, regime_detector, economic_data, news_data, returns_data)
    else:
        # Load saved allocation
        trend_predictor.predict_trends()
        trend_predictor.adjust_for_economic_uncertainty(economic_data)
        trend_predictor.adjust_for_news_sentiment(news_data)
        
        portfolio_allocator = BalancedPortfolioAllocator(
            trend_predictor.final_direction,
            trend_predictor.final_strength,
            regime=regime_detector.get_current_regime()
        )
        portfolio_allocator.load_allocation()
    
    if args.mode in ['backtest', 'run_all']:
        # Step 5: Run backtest
        backtest_engine = run_backtest(
            price_data,
            returns_data,
            regimes=regime_detector.named_regimes,
            economic_data=economic_data
        )
        
        # Print summary
        print_summary(portfolio_allocator, backtest_engine)
    
    print("\n" + "*"*80)
    print("*" + " "*30 + "STRATEGY EXECUTION COMPLETE" + " "*27 + "*")
    print("*"*80 + "\n")

if __name__ == "__main__":
    main()