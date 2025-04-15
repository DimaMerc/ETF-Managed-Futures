"""
Trend prediction module for the Enhanced Managed Futures ETF Strategy.
Uses LSTM networks and technical indicators to predict price trends across different asset classes.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as keras
import keras
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config
import utils

class TrendPredictor:
    """Class for predicting trends using LSTM models and technical indicators."""
    
    def __init__(self, returns_data, technical_indicators=None, sequence_length=None, epochs=None):
        """
        Initialize the TrendPredictor.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Asset returns data
        technical_indicators : dict, optional
            Dictionary of technical indicators for each asset
        sequence_length : int, optional
            Sequence length for LSTM models (default: from config)
        epochs : int, optional
            Number of epochs for training LSTM models (default: from config)
        """
        self.returns = returns_data
        self.technical_indicators = technical_indicators
        self.sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
        self.epochs = epochs or config.LSTM_EPOCHS
        
        # Initialize models
        self.trend_models = {}
        self.scalers = {}
        
        # Store predictions
        self.trend_predictions = None
        self.trend_direction = None
        self.trend_strength = None
        
        # Create models directory
        utils.create_directory('models')
    
    # Replace this method in trend_predictor.py

    def build_lstm_model(self, input_shape):
        """
        Build an LSTM model for trend prediction with proper Keras syntax.
        
        Parameters:
        -----------
        input_shape : tuple
            Input shape for the model (sequence_length, features)
        
        Returns:
        --------
        keras.Model
            LSTM model
        """
        from tensorflow import keras
        
        # Create model using the proper functional API approach
        inputs = keras.layers.Input(shape=input_shape)
        
        # Bidirectional LSTM layer with return sequences
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, activation='relu', return_sequences=True))(inputs)
        x = keras.layers.Dropout(0.2)(x)
        
        # Second LSTM layer
        x = keras.layers.LSTM(32, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = keras.layers.Dense(1)(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def prepare_lstm_data(self, series, seq_length):
        """
        Prepare data for LSTM model.
        
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
    
    def build_trend_models(self):
        """
        Build LSTM-based trend prediction models for each asset.
        
        Returns:
        --------
        dict
            Dictionary of trend models
        """
        print("Building trend prediction models...")
        
        returns = self.returns
        
        # Build a model for each asset
        for ticker in returns.columns:
            print(f"Building model for {ticker}...")
            
            # Prepare data
            series = returns[ticker].dropna().values
            if len(series) <= self.sequence_length:
                print(f"Not enough data for {ticker}")
                continue
                
            X, y = self.prepare_lstm_data(series, self.sequence_length)
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_flat = X.reshape(-1, 1)
            X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
            y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
            
            # Store scaler for later use
            self.scalers[ticker] = scaler
            
            # Reshape for LSTM [samples, time steps, features]
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            # Split into train and validation
            split = int(0.8 * len(X_scaled))
            X_train, X_val = X_scaled[:split], X_scaled[split:]
            y_train, y_val = y_scaled[:split], y_scaled[split:]
            
            # Build LSTM model
            model = self.build_lstm_model((self.sequence_length, 1))
            
            # Train model with early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Store model
            self.trend_models[ticker] = {
                'model': model,
                'sequence_length': self.sequence_length
            }
        
        print(f"Built trend models for {len(self.trend_models)} assets")
        return self.trend_models
    
    def save_trend_models(self):
        """
        Save trend prediction models to disk.
        
        Returns:
        --------
        None
        """
        print("Saving trend prediction models...")
        
        # Create models directory
        utils.create_directory('models')
        
        for ticker, model_info in self.trend_models.items():
            # Create directory for each asset
            utils.create_directory(f'models/{ticker}')
            
            # Save model with .keras extension (required in newer TensorFlow versions)
            model_info['model'].save(f'models/{ticker}/lstm_model.keras')
            
            # Save scaler
            if ticker in self.scalers:
                with open(f'models/{ticker}/scaler.pkl', 'wb') as f:
                    pickle.dump(self.scalers[ticker], f)
            
            # Save metadata
            metadata = {
                'sequence_length': model_info['sequence_length']
            }
            with open(f'models/{ticker}/metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
        
        print("Models saved successfully.")

    def load_trend_models(self):
        """
        Load trend prediction models from disk.
        
        Returns:
        --------
        dict
            Dictionary of trend models
        """
        print("Loading trend prediction models...")
        
        self.trend_models = {}
        self.scalers = {}
        
        # Check if models directory exists
        if not os.path.exists('models'):
            print("Models directory not found. Build models first.")
            return None
        
        # Load models for each asset
        for asset_dir in os.listdir('models'):
            if os.path.isdir(f'models/{asset_dir}'):
                ticker = asset_dir
                
                # Check if model files exist - try both .keras and old formats
                model_path = None
                for ext in ['.keras', '.h5', '']:  # Try different extensions
                    test_path = f'models/{ticker}/lstm_model{ext}'
                    if os.path.exists(test_path):
                        model_path = test_path
                        break
                
                if not model_path:
                    print(f"No model file found for {ticker}")
                    continue
                
                try:
                    # Load model
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                    
                    # Load metadata
                    if os.path.exists(f'models/{ticker}/metadata.pkl'):
                        with open(f'models/{ticker}/metadata.pkl', 'rb') as f:
                            metadata = pickle.load(f)
                        
                        sequence_length = metadata.get('sequence_length', self.sequence_length)
                    else:
                        sequence_length = self.sequence_length
                    
                    # Load scaler
                    if os.path.exists(f'models/{ticker}/scaler.pkl'):
                        with open(f'models/{ticker}/scaler.pkl', 'rb') as f:
                            self.scalers[ticker] = pickle.load(f)
                    
                    # Store model
                    self.trend_models[ticker] = {
                        'model': model,
                        'sequence_length': sequence_length
                    }
                    
                    print(f"Loaded model for {ticker}")
                except Exception as e:
                    print(f"Error loading model for {ticker}: {str(e)}")
        
        print(f"Loaded {len(self.trend_models)} trend models")
        return self.trend_models
    
    def predict_trends(self):
        """
        Generate trend predictions for all assets using LSTM models and technical indicators.
        
        Returns:
        --------
        tuple
            (trend_predictions, trend_direction, trend_strength)
        """
        print("Generating trend predictions...")
        
        returns = self.returns
        
        # Check if we have models
        if not self.trend_models:
            print("No trend models available. Build or load models first.")
            return None, None, None
        
        predictions = {}
        tech_predictions = {}
        
        # 1. Get LSTM model predictions
        for ticker, model_info in self.trend_models.items():
            model = model_info['model']
            seq_length = model_info['sequence_length']
            
            # Check if we have enough data
            if ticker not in returns.columns or len(returns[ticker]) < seq_length:
                continue
            
            # Get the most recent sequence
            last_sequence = returns[ticker].values[-seq_length:]
            
            # Check if we have any NaNs
            if np.isnan(last_sequence).any():
                print(f"Warning: NaN values in sequence for {ticker}")
                continue
            
            # Scale the sequence if we have a scaler
            if ticker in self.scalers:
                scaler = self.scalers[ticker]
                last_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
            
            # Reshape for LSTM prediction
            X = last_sequence.reshape(1, seq_length, 1)
            
            # Make prediction
            pred = model.predict(X, verbose=0)[0][0]
            
            # Inverse transform if we have a scaler
            if ticker in self.scalers:
                pred = self.scalers[ticker].inverse_transform(np.array([[pred]]))[0][0]
            
            predictions[ticker] = pred
        
        # 2. Get predictions from technical indicators
        if self.technical_indicators:
            for ticker, indicators in self.technical_indicators.items():
                # Initialize signal
                signal = 0
                signal_count = 0
                
                # Process each indicator
                for indicator_name, indicator_df in indicators.items():
                    if indicator_df.empty:
                        continue
                    
                    latest_indicator = indicator_df.iloc[-1]
                    
                    if indicator_name == 'SMA':
                        # Compare price to SMA
                        latest_price = returns[ticker].iloc[-self.sequence_length:].mean()  # Approximate
                        sma = latest_indicator.get('SMA', 0)
                        # Positive if price > SMA
                        if pd.notnull(latest_price) and pd.notnull(sma):
                            signal += 1 if latest_price > sma else -1
                            signal_count += 1
                    
                    elif indicator_name == 'EMA':
                        # Compare price to EMA
                        latest_price = returns[ticker].iloc[-self.sequence_length:].mean()
                        ema = latest_indicator.get('EMA', 0)
                        # Positive if price > EMA
                        if pd.notnull(latest_price) and pd.notnull(ema):
                            signal += 1 if latest_price > ema else -1
                            signal_count += 1
                    
                    elif indicator_name == 'RSI':
                        # RSI interpretation
                        rsi = latest_indicator.get('RSI', 50)
                        if pd.notnull(rsi):
                            # Oversold if RSI < 30, overbought if RSI > 70
                            if rsi < 30:
                                signal += 1  # Bullish
                            elif rsi > 70:
                                signal += -1  # Bearish
                            signal_count += 1
                    
                    elif indicator_name == 'STOCH':
                        # Stochastic interpretation
                        k = latest_indicator.get('SlowK', 50)
                        d = latest_indicator.get('SlowD', 50)
                        if pd.notnull(k) and pd.notnull(d):
                            # Oversold if both < 20, overbought if both > 80
                            if k < 20 and d < 20:
                                signal += 1  # Bullish
                            elif k > 80 and d > 80:
                                signal += -1  # Bearish
                            # K crossing above D is bullish
                            elif k > d:
                                signal += 0.5
                            # K crossing below D is bearish
                            elif k < d:
                                signal += -0.5
                            signal_count += 1
                    
                    elif indicator_name == 'ADX':
                        # ADX interpretation (trend strength only)
                        adx = latest_indicator.get('ADX', 15)
                        if pd.notnull(adx):
                            # Strong trend if ADX > 25, we don't know direction from ADX alone
                            # Just add a small weight to existing signal
                            if adx > 25 and signal != 0:
                                signal *= 1.2
                            signal_count += 0.2  # Lower weight for ADX
                
                # Average the signals
                if signal_count > 0:
                    tech_predictions[ticker] = signal / signal_count
        
        # 3. Combine LSTM and technical indicator predictions (if available)
        combined_predictions = {}
        for ticker in set(list(predictions.keys()) + list(tech_predictions.keys())):
            lstm_pred = predictions.get(ticker, 0)
            tech_pred = tech_predictions.get(ticker, 0)
            
            # Weight LSTM predictions higher (0.7) than technical indicators (0.3)
            if ticker in predictions and ticker in tech_predictions:
                combined_predictions[ticker] = 0.7 * lstm_pred + 0.3 * tech_pred
            elif ticker in predictions:
                combined_predictions[ticker] = lstm_pred
            else:
                combined_predictions[ticker] = tech_pred
        
        # Convert to Series
        self.trend_predictions = pd.Series(combined_predictions)
        
        # Calculate prediction strength (absolute value of predictions)
        self.trend_strength = self.trend_predictions.abs()
        
        # Determine direction (1 for up, -1 for down)
        self.trend_direction = np.sign(self.trend_predictions)
        
        print(f"Generated trend predictions for {len(combined_predictions)} assets")
        return self.trend_predictions, self.trend_direction, self.trend_strength
    
    def adjust_for_economic_uncertainty(self, economic_data):
        """
        Adjust trend signals based on economic uncertainty and tariff impact.
        
        Parameters:
        -----------
        economic_data : pd.DataFrame
            Economic indicators data
        
        Returns:
        --------
        tuple
            (adjusted_direction, adjusted_strength)
        """
        print("Adjusting for economic uncertainty...")
        
        # Check if we have predictions
        if self.trend_direction is None or self.trend_strength is None:
            print("No trend predictions available. Run predict_trends() first.")
            return None, None
        
        # Get the most recent economic data
        if economic_data is None or economic_data.empty:
            print("No economic data available. Skipping adjustment.")
            return self.trend_direction, self.trend_strength
        
        # Get the most recent economic data
        latest_econ = economic_data.iloc[-1]
        
        # Extract uncertainty and tariff measures
        uncertainty = latest_econ.get('EPU_Index', 50)
        tariff_impact = latest_econ.get('Tariff_Impact', 50)
        
        # Adjust trend strengths based on uncertainty and tariffs
        adjusted_strength = self.trend_strength.copy()
        adjusted_direction = self.trend_direction.copy()
        
        # Logic for commodities - tariffs often increase domestic commodity prices
        for ticker in config.COMMODITY_FUTURES:
            if ticker in adjusted_strength.index:
                # Increase trend strength for domestic commodities under tariffs
                adjusted_strength[ticker] *= (1 + 0.2 * tariff_impact / 100)
                
                # For commodities affected by tariffs, potentially reinforce upward trends
                if 'OIL' in ticker or 'COPPER' in ticker or 'ALUMINUM' in ticker:
                    if adjusted_direction[ticker] > 0:
                        adjusted_strength[ticker] *= 1.2  # Strengthen up trends
        
        # Logic for currencies - tariffs often affect currency relationships
        for ticker in config.CURRENCY_FUTURES:
            if ticker in adjusted_strength.index:
                # Dollar might strengthen with tariffs
                if ticker == 'EUR':  # Euro
                    if tariff_impact > 70:
                        adjusted_direction[ticker] = -1  # Likely Euro weakness vs USD
                
                # Increase volatility of currency predictions under uncertainty
                adjusted_strength[ticker] *= (1 + 0.15 * uncertainty / 100)
        
        # Logic for bonds - flight to safety during high uncertainty
        for ticker in config.BOND_FUTURES:
            if ticker in adjusted_strength.index:
                # During high uncertainty, strengthen trend for safe-haven assets
                if uncertainty > 70:
                    # For Treasury Yield, negative direction means bond prices go up
                    # (yields down, prices up)
                    adjusted_direction[ticker] = -1  # Favor upward bond price trend
                    adjusted_strength[ticker] *= 1.3
        
        # Logic for equities - increased uncertainty typically negative
        for ticker in config.EQUITY_FUTURES:
            if ticker in adjusted_strength.index:
                # Higher uncertainty typically weakens equity markets
                if uncertainty > 70:
                    adjusted_direction[ticker] = -1  # Bearish on equities during high uncertainty
                    adjusted_strength[ticker] *= 1.2
                
                # Trade-dependent sectors may be more affected by tariffs
                if ticker in ['QQQ', 'SPY'] and tariff_impact > 70:
                    adjusted_direction[ticker] = -1
                    adjusted_strength[ticker] *= 1.3
        
        # Store adjusted predictions
        self.adjusted_direction = adjusted_direction
        self.adjusted_strength = adjusted_strength
        
        print("Adjusted trend predictions for economic uncertainty")
        return self.adjusted_direction, self.adjusted_strength
    
    def adjust_for_news_sentiment(self, news_sentiment):
        """
        Further adjust trend signals based on recent news sentiment.
        
        Parameters:
        -----------
        news_sentiment : pd.DataFrame
            News sentiment data
        
        Returns:
        --------
        tuple
            (final_direction, final_strength)
        """
        print("Adjusting for news sentiment...")
        
        # Check if we have adjusted predictions
        if not hasattr(self, 'adjusted_direction') or not hasattr(self, 'adjusted_strength'):
            print("No adjusted predictions available. Run adjust_for_economic_uncertainty() first.")
            return self.trend_direction, self.trend_strength
        
        # Check if we have news sentiment data
        if news_sentiment is None or news_sentiment.empty:
            print("No news sentiment data available. Skipping adjustment.")
            return self.adjusted_direction, self.adjusted_strength
        
        # Get the latest sentiment
        latest_sentiment = news_sentiment['aggregate_sentiment'].iloc[-1]
        
        # Start with the economically adjusted values
        final_direction = self.adjusted_direction.copy()
        final_strength = self.adjusted_strength.copy()
        
        # If sentiment is strongly negative about tariffs/trade
        if latest_sentiment < -0.3:
            print("News sentiment is strongly negative - adjusting predictions")
            
            # Strengthen negative trends for assets sensitive to trade tensions
            for ticker in list(config.EQUITY_FUTURES.keys()) + list(config.CURRENCY_FUTURES.keys()):
                if ticker in final_direction.index:
                    if final_direction[ticker] < 0:
                        final_strength[ticker] *= 1.2  # Strengthen negative trends
            
            # Safe haven assets might strengthen
            safe_havens = []
            if 'TREASURY_YIELD' in config.BOND_FUTURES:
                safe_havens.append('TREASURY_YIELD')
            
            for ticker in safe_havens:
                if ticker in final_direction.index:
                    # For Treasury Yield, negative direction means bond prices go up
                    final_direction[ticker] = -1  # Favor upward bond price trend (yield down)
                    final_strength[ticker] *= 1.1
        
        # If sentiment is strongly positive
        elif latest_sentiment > 0.3:
            print("News sentiment is strongly positive - adjusting predictions")
            
            # Strengthen positive trends for risk assets
            for ticker in list(config.EQUITY_FUTURES.keys()):
                if ticker in final_direction.index:
                    if final_direction[ticker] > 0:
                        final_strength[ticker] *= 1.2  # Strengthen positive trends
            
            # Reduce emphasis on safe haven assets
            safe_havens = []
            if 'TREASURY_YIELD' in config.BOND_FUTURES:
                safe_havens.append('TREASURY_YIELD')
            
            for ticker in safe_havens:
                if ticker in final_direction.index:
                    final_strength[ticker] *= 0.9
        
        # Store final adjusted predictions
        self.final_direction = final_direction
        self.final_strength = final_strength
        
        print("Final trend predictions adjusted for news sentiment")
        return self.final_direction, self.final_strength
    
    def save_predictions(self, filepath='data/predictions.csv'):
        """
        Save predictions to a CSV file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the predictions (default: 'data/predictions.csv')
        
        Returns:
        --------
        None
        """
        if hasattr(self, 'final_direction') and hasattr(self, 'final_strength'):
            predictions_df = pd.DataFrame({
                'trend_prediction': self.trend_predictions,
                'trend_direction': self.trend_direction,
                'trend_strength': self.trend_strength,
                'adjusted_direction': self.adjusted_direction,
                'adjusted_strength': self.adjusted_strength,
                'final_direction': self.final_direction,
                'final_strength': self.final_strength
            })
        elif hasattr(self, 'trend_direction') and hasattr(self, 'trend_strength'):
            predictions_df = pd.DataFrame({
                'trend_prediction': self.trend_predictions,
                'trend_direction': self.trend_direction,
                'trend_strength': self.trend_strength
            })
        else:
            print("No predictions to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        predictions_df.to_csv(filepath)
        print(f"Predictions saved to {filepath}")
    
    def load_predictions(self, filepath='data/predictions.csv'):
        """
        Load predictions from a CSV file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to load the predictions from (default: 'data/predictions.csv')
        
        Returns:
        --------
        pd.DataFrame
            Loaded predictions
        """
        try:
            predictions_df = pd.read_csv(filepath, index_col=0)
            
            # Load predictions
            self.trend_predictions = predictions_df['trend_prediction']
            self.trend_direction = predictions_df['trend_direction']
            self.trend_strength = predictions_df['trend_strength']
            
            # Load adjusted predictions if available
            if 'adjusted_direction' in predictions_df.columns:
                self.adjusted_direction = predictions_df['adjusted_direction']
                self.adjusted_strength = predictions_df['adjusted_strength']
            
            # Load final predictions if available
            if 'final_direction' in predictions_df.columns:
                self.final_direction = predictions_df['final_direction']
                self.final_strength = predictions_df['final_strength']
            
            print(f"Predictions loaded from {filepath}")
            return predictions_df
        except FileNotFoundError:
            print(f"Predictions file {filepath} not found.")
            return None