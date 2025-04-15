"""
Dynamic volatility targeting module for risk management in the ETF strategy.
Adjusts portfolio leverage based on market conditions and volatility forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class VolatilityTargeting:
    """
    Class for dynamic volatility targeting and risk management.
    Adjusts portfolio leverage based on volatility forecasts.
    """
    
    def __init__(self, returns_data, target_volatility=0.10, 
                 max_leverage=2.0, min_leverage=0.5, lookback_window=60,
                 vol_forecast_horizon=20, current_regime=None):
        """
        Initialize the VolatilityTargeting.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data for all assets
        target_volatility : float, optional
            Target annualized volatility (default: 0.10 = 10%)
        max_leverage : float, optional
            Maximum allowed leverage (default: 2.0)
        min_leverage : float, optional
            Minimum allowed leverage (default: 0.5)
        lookback_window : int, optional
            Lookback window for volatility estimation (default: 60 days)
        vol_forecast_horizon : int, optional
            Horizon for volatility forecasts (default: 20 days)
        current_regime : str, optional
            Current market regime (default: None)
        """
        self.returns = returns_data
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.lookback_window = lookback_window
        self.vol_forecast_horizon = vol_forecast_horizon
        self.current_regime = current_regime
        
        # Initialize components
        self.historical_vol = None
        self.realized_vol = None
        self.predicted_vol = None
        self.vol_model = None
        self.current_leverage = 1.0
        self.leverage_history = None
    
    def calculate_historical_volatility(self):
        """
        Calculate historical realized volatility.
        
        Returns:
        --------
        pd.Series
            Historical realized volatility
        """
        # Calculate returns volatility for each asset
        asset_vols = self.returns.rolling(window=self.lookback_window).std() * np.sqrt(252)
        
        # Calculate portfolio volatility (simplified as average asset volatility)
        # In a real implementation, this would consider correlations and weights
        portfolio_vol = asset_vols.mean(axis=1)
        
        self.historical_vol = portfolio_vol
        return portfolio_vol
    
    def build_volatility_forecast_model(self):
        """
        Build a model to forecast volatility.
        
        Returns:
        --------
        object
            Trained volatility forecast model
        """
        print("Building volatility forecast model...")
        
        if self.historical_vol is None:
            self.calculate_historical_volatility()
        
        try:
            # Create volatility features
            features = pd.DataFrame(index=self.historical_vol.index)
            
            # Feature 1: Current volatility levels
            for lag in [1, 5, 10, 20, 40]:
                if lag < len(self.historical_vol):
                    features[f'vol_lag_{lag}'] = self.historical_vol.shift(lag)
            
            # Feature 2: Volatility ratios (trends in volatility)
            if len(self.historical_vol) > 20:
                features['vol_ratio_5_20'] = (
                    self.historical_vol.shift(1).rolling(window=5).mean() / 
                    self.historical_vol.shift(1).rolling(window=20).mean()
                )
            
            # Feature 3: Return features
            if self.returns is not None:
                # Calculate average absolute returns
                abs_returns = self.returns.abs().mean(axis=1)
                features['abs_ret_5d'] = abs_returns.rolling(window=5).mean()
                features['abs_ret_20d'] = abs_returns.rolling(window=20).mean()
                
                # Calculate return ratio
                features['ret_ratio_5_20'] = (
                    abs_returns.rolling(window=5).mean() / 
                    abs_returns.rolling(window=20).mean()
                )
                
                # Calculate drawdowns
                portfolio_returns = self.returns.mean(axis=1)
                wealth_index = (1 + portfolio_returns).cumprod()
                rolling_max = wealth_index.rolling(window=60, min_periods=1).max()
                drawdowns = (wealth_index - rolling_max) / rolling_max
                features['drawdown'] = drawdowns
            
            # Drop rows with NaN values
            features = features.dropna()
            
            # Prepare target: future volatility
            if len(features) > self.vol_forecast_horizon:
                # Target is average volatility over forecast horizon
                target = self.historical_vol.shift(-self.vol_forecast_horizon).rolling(
                    window=self.vol_forecast_horizon).mean()
                
                # Align with features
                target = target.loc[features.index]
                
                # Drop NaN values
                valid_indices = ~target.isna()
                features = features.loc[valid_indices]
                target = target.loc[valid_indices]
                
                if len(features) > 10:  # Ensure we have enough data to train
                    # Scale features
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    # Train a RandomForest model
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=5, 
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(features_scaled, target)
                    
                    # Store model and features for prediction
                    self.vol_model = {
                        'model': model,
                        'scaler': scaler,
                        'feature_names': features.columns.tolist()
                    }
                    
                    # Calculate model performance
                    predicted = model.predict(features_scaled)
                    mse = ((predicted - target) ** 2).mean()
                    rmse = np.sqrt(mse)
                    mean_target = target.mean()
                    
                    print(f"Volatility forecast model trained - RMSE: {rmse:.4f}, Mean target: {mean_target:.4f}")
                    
                    return self.vol_model
                else:
                    print("Not enough valid data points to train model")
                    return None
            else:
                print("Not enough historical data points for volatility forecasting")
                return None
                
        except Exception as e:
            print(f"Error building volatility forecast model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_volatility(self):
        """
        Predict future volatility using the trained model.
        
        Returns:
        --------
        float
            Predicted volatility
        """
        if self.vol_model is None:
            self.build_volatility_forecast_model()
            
        if self.vol_model is None:
            # Fall back to historical volatility if we couldn't build a model
            last_vol = self.historical_vol.iloc[-1] if self.historical_vol is not None else 0.15
            print(f"Using historical volatility as prediction: {last_vol:.4f}")
            self.predicted_vol = last_vol
            return last_vol
        
        try:
            # Create features for prediction
            features = pd.DataFrame(index=[0])
            
            # Use same feature creation logic as in model building
            for feature_name in self.vol_model['feature_names']:
                if feature_name.startswith('vol_lag_'):
                    lag = int(feature_name.split('_')[-1])
                    if len(self.historical_vol) > lag:
                        features[feature_name] = self.historical_vol.iloc[-lag]
                    else:
                        features[feature_name] = self.historical_vol.iloc[0]
                
                elif feature_name == 'vol_ratio_5_20':
                    if len(self.historical_vol) > 20:
                        vol_5d = self.historical_vol.iloc[-5:].mean()
                        vol_20d = self.historical_vol.iloc[-20:].mean()
                        ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
                        features[feature_name] = ratio
                    else:
                        features[feature_name] = 1.0
                
                elif feature_name.startswith('abs_ret_'):
                    window = int(feature_name.split('_')[-1][:-1])  # Extract number from 'abs_ret_5d'
                    if len(self.returns) > window:
                        abs_returns = self.returns.iloc[-window:].abs().mean(axis=1).mean()
                        features[feature_name] = abs_returns
                    else:
                        features[feature_name] = self.returns.abs().mean().mean()
                
                elif feature_name == 'ret_ratio_5_20':
                    if len(self.returns) > 20:
                        abs_ret_5d = self.returns.iloc[-5:].abs().mean(axis=1).mean()
                        abs_ret_20d = self.returns.iloc[-20:].abs().mean(axis=1).mean()
                        ratio = abs_ret_5d / abs_ret_20d if abs_ret_20d > 0 else 1.0
                        features[feature_name] = ratio
                    else:
                        features[feature_name] = 1.0
                
                elif feature_name == 'drawdown':
                    if len(self.returns) > 1:
                        portfolio_returns = self.returns.iloc[-60:].mean(axis=1)
                        wealth_index = (1 + portfolio_returns).cumprod()
                        peak = wealth_index.max()
                        current = wealth_index.iloc[-1]
                        drawdown = (current - peak) / peak if peak > 0 else 0
                        features[feature_name] = drawdown
                    else:
                        features[feature_name] = 0.0
            
            # Scale features
            features_scaled = self.vol_model['scaler'].transform(features)
            
            # Make prediction
            predicted_vol = self.vol_model['model'].predict(features_scaled)[0]
            
            # Apply sanity checks - volatility shouldn't deviate too much from recent history
            if predicted_vol < 0.01:  # Minimum 1% volatility
                predicted_vol = 0.01
            
            if predicted_vol > 0.50:  # Cap at 50% volatility
                predicted_vol = 0.50
            
            recent_vol = self.historical_vol.iloc[-20:].mean() if len(self.historical_vol) >= 20 else self.historical_vol.mean()
            
            # Don't allow prediction to be less than half or more than twice recent volatility
            predicted_vol = max(predicted_vol, recent_vol * 0.5)
            predicted_vol = min(predicted_vol, recent_vol * 2.0)
            
            print(f"Predicted volatility: {predicted_vol:.4f}, Recent historical: {recent_vol:.4f}")
            
            self.predicted_vol = predicted_vol
            return predicted_vol
            
        except Exception as e:
            print(f"Error predicting volatility: {str(e)}")
            
            # Fall back to recent historical volatility
            if self.historical_vol is not None and len(self.historical_vol) > 0:
                recent_vol = self.historical_vol.iloc[-1]
                print(f"Using recent historical volatility: {recent_vol:.4f}")
                self.predicted_vol = recent_vol
                return recent_vol
            else:
                # Default volatility if all else fails
                print("Using default volatility of 15%")
                self.predicted_vol = 0.15
                return 0.15
    
    def calculate_target_leverage(self):
        """
        Calculate target leverage based on volatility forecast and regime.
        
        Returns:
        --------
        float
            Target leverage
        """
        if self.predicted_vol is None:
            self.predict_volatility()
        
        # Base leverage calculation using the volatility targeting formula
        base_leverage = self.target_volatility / self.predicted_vol
        
        # Apply leverage constraints
        base_leverage = max(min(base_leverage, self.max_leverage), self.min_leverage)
        
        # Adjust leverage based on current regime if available
        adjusted_leverage = base_leverage
        
        if self.current_regime is not None:
            if "Crisis" in self.current_regime:
                # Reduce leverage in crisis regimes
                adjusted_leverage = min(base_leverage, 0.75)
                print(f"Crisis regime: Reducing leverage from {base_leverage:.2f} to {adjusted_leverage:.2f}")
            
            elif "High Uncertainty" in self.current_regime or "Trade Tension" in self.current_regime:
                # Reduce leverage in uncertain regimes
                adjusted_leverage = min(base_leverage, 1.0)
                print(f"Uncertainty regime: Reducing leverage from {base_leverage:.2f} to {adjusted_leverage:.2f}")
            
            elif "Risk-On" in self.current_regime:
                # Potentially increase leverage in risk-on regimes, but still respect volatility target
                adjusted_leverage = min(base_leverage * 1.1, self.max_leverage)
                print(f"Risk-on regime: Adjusting leverage from {base_leverage:.2f} to {adjusted_leverage:.2f}")
        
        # Apply final leverage constraints
        final_leverage = max(min(adjusted_leverage, self.max_leverage), self.min_leverage)
        
        # Print summary
        print(f"Target volatility: {self.target_volatility:.2%}")
        print(f"Predicted volatility: {self.predicted_vol:.2%}")
        print(f"Base leverage: {base_leverage:.2f}")
        print(f"Regime-adjusted leverage: {adjusted_leverage:.2f}")
        print(f"Final leverage: {final_leverage:.2f}")
        
        self.current_leverage = final_leverage
        
        # Update leverage history
        if self.leverage_history is None:
            self.leverage_history = pd.Series([final_leverage], index=[pd.Timestamp.now()])
        else:
            self.leverage_history = self.leverage_history.append(
                pd.Series([final_leverage], index=[pd.Timestamp.now()]))
        
        return final_leverage
    
    def adjust_portfolio_weights(self, weights):
        """
        Adjust portfolio weights based on target leverage.
        
        Parameters:
        -----------
        weights : pd.Series
            Current portfolio weights
        
        Returns:
        --------
        pd.Series
            Adjusted portfolio weights
        """
        if self.current_leverage is None:
            self.calculate_target_leverage()
        
        # Extract original exposure
        original_leverage = abs(weights).sum()
        
        # Scale weights to match target leverage
        if original_leverage > 0:
            scaling_factor = self.current_leverage / original_leverage
            adjusted_weights = weights * scaling_factor
        else:
            # If no exposure, distribute leverage evenly (should not happen in practice)
            print("Warning: No exposure in original weights.")
            adjusted_weights = weights
        
        return adjusted_weights
    
    def run_backtest_volatility_targeting(self, portfolio_weights_history, verbose=True):
        """
        Run a backtest of the volatility targeting strategy.
        
        Parameters:
        -----------
        portfolio_weights_history : pd.DataFrame
            Historical portfolio weights (assets in columns, time in index)
        verbose : bool, optional
            Whether to print detailed output (default: True)
        
        Returns:
        --------
        pd.DataFrame
            Backtest results with volatility-targeted weights
        """
        if verbose:
            print("Running volatility targeting backtest...")
        
        if self.historical_vol is None:
            self.calculate_historical_volatility()
        
        # Initialize results
        results = pd.DataFrame(index=portfolio_weights_history.index)
        results['original_leverage'] = portfolio_weights_history.abs().sum(axis=1)
        
        # Initialize adjusted weights DataFrame
        adjusted_weights = pd.DataFrame(index=portfolio_weights_history.index, 
                                        columns=portfolio_weights_history.columns)
        
        # Calculate target leverage for each period
        leverage_targets = pd.Series(index=portfolio_weights_history.index)
        realized_vols = pd.Series(index=portfolio_weights_history.index)
        
        # Use a rolling window for prediction
        for i, date in enumerate(portfolio_weights_history.index):
            # Skip the first window where we don't have enough history
            if i < self.lookback_window:
                leverage_targets[date] = 1.0
                adjusted_weights.loc[date] = portfolio_weights_history.loc[date]
                continue
            
            # Use data up to current date for prediction
            current_returns = self.returns.loc[:date]
            vol_targeter = VolatilityTargeting(
                returns_data=current_returns,
                target_volatility=self.target_volatility,
                max_leverage=self.max_leverage,
                min_leverage=self.min_leverage,
                lookback_window=self.lookback_window,
                vol_forecast_horizon=self.vol_forecast_horizon
            )
            
            # Calculate historical volatility and predict future volatility
            vol_targeter.calculate_historical_volatility()
            
            # For efficiency in backtest, use historical volatility instead of building a model
            realized_vol = vol_targeter.historical_vol.iloc[-1]
            realized_vols[date] = realized_vol
            
            # Calculate target leverage using volatility targeting formula
            target_leverage = vol_targeter.target_volatility / realized_vol
            target_leverage = max(min(target_leverage, vol_targeter.max_leverage), vol_targeter.min_leverage)
            leverage_targets[date] = target_leverage
            
            # Adjust weights
            original_weights = portfolio_weights_history.loc[date]
            original_leverage = abs(original_weights).sum()
            
            if original_leverage > 0:
                scaling_factor = target_leverage / original_leverage
                adjusted_weights.loc[date] = original_weights * scaling_factor
            else:
                adjusted_weights.loc[date] = original_weights
        
        # Calculate portfolio returns and risk metrics
        results['target_leverage'] = leverage_targets
        results['realized_volatility'] = realized_vols
        
        if verbose:
            print("Volatility targeting backtest completed:")
            print(f"  Average original leverage: {results['original_leverage'].mean():.2f}")
            print(f"  Average target leverage: {results['target_leverage'].mean():.2f}")
            print(f"  Average realized volatility: {results['realized_volatility'].mean():.2%}")
        
        return results, adjusted_weights
    
    def visualize_volatility_targeting(self, backtest_results=None):
        """
        Visualize volatility targeting analysis.
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame, optional
            Results from volatility targeting backtest (default: None)
        
        Returns:
        --------
        None
        """
        if self.historical_vol is None:
            self.calculate_historical_volatility()
        
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Historical Volatility
        plt.subplot(3, 1, 1)
        
        if self.historical_vol is not None:
            self.historical_vol.plot(label='Historical Volatility')
            plt.axhline(y=self.target_volatility, color='r', linestyle='--', 
                       label=f'Target Volatility ({self.target_volatility:.1%})')
            plt.title('Historical Portfolio Volatility')
            plt.ylabel('Annualized Volatility')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No historical volatility data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 2: Leverage Analysis
        if backtest_results is not None:
            plt.subplot(3, 1, 2)
            
            backtest_results['original_leverage'].plot(label='Original Leverage', alpha=0.7)
            backtest_results['target_leverage'].plot(label='Target Leverage', alpha=0.7)
            
            plt.title('Leverage Analysis')
            plt.ylabel('Leverage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot regime overlays if available
            if 'regime' in backtest_results.columns:
                unique_regimes = backtest_results['regime'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
                
                for i, regime in enumerate(unique_regimes):
                    regime_dates = backtest_results[backtest_results['regime'] == regime].index
                    if not regime_dates.empty:
                        plt.fill_between(regime_dates, 0, 1, 
                                         transform=plt.gca().get_xaxis_transform(),
                                         color=colors[i], alpha=0.2, label=regime)
                
                plt.legend(loc='upper left')
            
            # Plot 3: Scatter plot of volatility vs leverage
            plt.subplot(3, 1, 3)
            
            plt.scatter(backtest_results['realized_volatility'], 
                       backtest_results['target_leverage'],
                       alpha=0.5)
            
            # Add a trend line
            from scipy import stats
            if len(backtest_results) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(
                    backtest_results['realized_volatility'], 
                    backtest_results['target_leverage'])
                
                x = np.array([backtest_results['realized_volatility'].min(),
                              backtest_results['realized_volatility'].max()])
                y = intercept + slope * x
                plt.plot(x, y, 'r--', label=f'Trend (r={r_value:.2f})')
            
            # Add the theoretical volatility targeting curve
            x = np.linspace(0.01, backtest_results['realized_volatility'].max() * 1.2, 100)
            y = self.target_volatility / x
            y = np.clip(y, self.min_leverage, self.max_leverage)
            plt.plot(x, y, 'g-', label='Volatility Targeting Formula')
            
            plt.title('Relationship Between Volatility and Leverage')
            plt.xlabel('Realized Volatility')
            plt.ylabel('Target Leverage')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Plot leverage predictions if we have them
            plt.subplot(3, 1, 2)
            
            if self.predicted_vol is not None:
                plt.bar(['Predicted Volatility', 'Target Volatility', 'Target Leverage'], 
                       [self.predicted_vol, self.target_volatility, self.current_leverage],
                       color=['skyblue', 'coral', 'lightgreen'])
                plt.title('Current Volatility and Leverage Analysis')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No volatility prediction data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
            
            # Plot 3: Historical leverage if available
            plt.subplot(3, 1, 3)
            
            if self.leverage_history is not None and len(self.leverage_history) > 1:
                self.leverage_history.plot()
                plt.title('Leverage History')
                plt.ylabel('Target Leverage')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No leverage history data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()