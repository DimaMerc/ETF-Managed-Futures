"""
Enhanced market regime detection module using Gaussian Mixture Models with robust error handling
and regime transition probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

class EnhancedRegimeDetector:
    """Class for detecting market regimes using GMM with transition probabilities."""
    
    def __init__(self, returns_data, economic_data=None, n_regimes=5, lookback=None):
        """
        Initialize the EnhancedRegimeDetector.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Asset returns data
        economic_data : pd.DataFrame, optional
            Economic indicators data (default: None)
        n_regimes : int, optional
            Number of regimes to detect (default: 5)
        lookback : int, optional
            Lookback window for feature calculation (default: min(60, len(returns_data)//4))
        """
        self.returns = returns_data
        self.economic_data = economic_data
        self.n_regimes = n_regimes
        self.lookback = lookback or min(60, len(returns_data)//4)
        
        # Initialize components
        self.regime_model = None
        self.regimes = None
        self.named_regimes = None
        self.regime_mapping = None
        self.regime_features = None
        self.regime_probs = None  # New: Store regime probabilities
        self.transition_matrix = None  # New: Store regime transition probabilities
        
        # Configure default asset classes (can be overridden later)
        self._setup_asset_classes()
    
    def _setup_asset_classes(self):
        """Setup asset class groupings based on returns columns."""
        # This is a simplified approach - in production, use a more robust method
        columns = self.returns.columns
        self.asset_classes = {
            'Commodities': [col for col in columns if any(item in col.upper() for item in 
                          ['WTI', 'BRENT', 'GAS', 'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE'])],
            'Currencies': [col for col in columns if any(item in col.upper() for item in 
                         ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF'])],
            'Bonds': [col for col in columns if any(item in col.upper() for item in 
                    ['TREASURY', 'BOND', 'YIELD'])],
            'Equities': [col for col in columns if any(item in col.upper() for item in 
                        ['SPY', 'QQQ', 'DIA', 'IWM', 'INDEX'])]
        }
        
        # Add any remaining columns to a 'Others' category
        assigned = [item for sublist in self.asset_classes.values() for item in sublist]
        remaining = [col for col in columns if col not in assigned]
        if remaining:
            self.asset_classes['Others'] = remaining
    
    def prepare_features(self):
        """
        Prepare features for regime detection with robust error handling.
        
        Returns:
        --------
        pd.DataFrame
            Features for regime detection
        """
        print("Preparing features for enhanced regime detection...")
        
        try:
            # Basic validation
            if self.returns is None or self.returns.empty:
                raise ValueError("No returns data available")
            
            returns = self.returns.copy()
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=returns.index)
            
            # 1. Asset class level statistics
            for asset_class, assets in self.asset_classes.items():
                if not assets or not set(assets).issubset(returns.columns):
                    continue
                
                # Get returns for this asset class
                class_returns = returns[assets]
                
                # Calculate volatility features
                try:
                    vol = class_returns.rolling(window=self.lookback).std()
                    features[f'{asset_class}_vol'] = vol.mean(axis=1)
                except Exception as e:
                    print(f"Error calculating volatility for {asset_class}: {str(e)}")
                
                # Calculate return features
                try:
                    for period in [5, 20, 60]:
                        if len(returns) > period:
                            rolling_ret = class_returns.rolling(window=period).mean()
                            features[f'{asset_class}_ret_{period}d'] = rolling_ret.mean(axis=1)
                except Exception as e:
                    print(f"Error calculating returns for {asset_class}: {str(e)}")
                
                # Calculate cross-asset correlations
                try:
                    if len(assets) > 1:
                        # Calculate pairwise correlations within asset class
                        corr = class_returns.rolling(window=self.lookback).corr()
                        # Extract unique pairs (excluding self-correlations)
                        corr_pairs = []
                        for i, asset1 in enumerate(assets):
                            for j, asset2 in enumerate(assets):
                                if i < j:  # Only include each pair once
                                    if (asset1 in corr.index.levels[0] and 
                                        asset2 in corr.index.levels[0]):
                                        try:
                                            # For each date, get the correlation between this pair
                                            pair_corr = []
                                            for date in features.index:
                                                try:
                                                    if date in corr.index.levels[0]:
                                                        val = corr.loc[(date, asset1), asset2]
                                                        if np.isfinite(val):
                                                            pair_corr.append(val)
                                                        else:
                                                            pair_corr.append(np.nan)
                                                    else:
                                                        pair_corr.append(np.nan)
                                                except Exception:
                                                    pair_corr.append(np.nan)
                                            
                                            # Add to features if we have enough valid values
                                            if len(pair_corr) == len(features) and not all(np.isnan(pair_corr)):
                                                features[f'{asset1}_{asset2}_corr'] = pair_corr
                                        except Exception as e:
                                            print(f"Error calculating correlation for {asset1}-{asset2}: {str(e)}")
                except Exception as e:
                    print(f"Error calculating correlations for {asset_class}: {str(e)}")
            
            # 2. Cross-asset class correlations
            # Calculate correlations between asset classes
            try:
                class_returns = {}
                for asset_class, assets in self.asset_classes.items():
                    if assets and set(assets).issubset(returns.columns):
                        # Use mean return as class return
                        class_returns[asset_class] = returns[assets].mean(axis=1)
                
                if len(class_returns) > 1:
                    class_df = pd.DataFrame(class_returns)
                    
                    # Calculate rolling correlations between classes
                    roll_corr = class_df.rolling(window=self.lookback).corr()
                    
                    # Extract pairwise correlations
                    for i, class1 in enumerate(class_df.columns):
                        for j, class2 in enumerate(class_df.columns):
                            if i < j:  # Only include each pair once
                                try:
                                    # For each date, get the correlation
                                    pair_corr = []
                                    for date in features.index:
                                        try:
                                            if date in roll_corr.index.levels[0]:
                                                val = roll_corr.loc[(date, class1), class2]
                                                if np.isfinite(val):
                                                    pair_corr.append(val)
                                                else:
                                                    pair_corr.append(np.nan)
                                            else:
                                                pair_corr.append(np.nan)
                                        except Exception:
                                            pair_corr.append(np.nan)
                                    
                                    # Add to features if we have enough valid values
                                    if len(pair_corr) == len(features) and not all(np.isnan(pair_corr)):
                                        features[f'{class1}_{class2}_corr'] = pair_corr
                                except Exception as e:
                                    print(f"Error calculating correlation for {class1}-{class2}: {str(e)}")
            except Exception as e:
                print(f"Error calculating cross-asset correlations: {str(e)}")
            
            # 3. Economic indicators (if available)
            if self.economic_data is not None and not self.economic_data.empty:
                try:
                    # Get indices that exist in both returns and economic data
                    common_idx = features.index.intersection(self.economic_data.index)
                    if len(common_idx) > 0:
                        for col in self.economic_data.columns:
                            # Get data and reindex to match features
                            econ_series = self.economic_data[col].reindex(common_idx)
                            # Extend to all dates using forward fill
                            econ_series = econ_series.reindex(features.index, method='ffill')
                            features[col] = econ_series
                except Exception as e:
                    print(f"Error incorporating economic data: {str(e)}")
            
            # Clean features - remove columns with too many NaNs
            features = features.dropna(axis=1, thresh=len(features)*0.7)  # Keep columns with at least 70% non-NaN values
            
            # Fill remaining NaNs with forward and backward fill
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Final check for any remaining issues
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Store for later use
            self.regime_features = features
            
            print(f"Prepared {features.shape[1]} features for regime detection with {len(features)} observations")
            return features
        
        except Exception as e:
            print(f"Error in feature preparation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create default features as fallback
            default_features = pd.DataFrame(index=self.returns.index)
            default_features['default'] = 0.0
            self.regime_features = default_features
            return default_features
    
    def detect_regimes(self):
        """
        Detect market regimes using Gaussian Mixture Models with robust error handling.
        
        Returns:
        --------
        tuple
            (regimes, regime_mapping, regime_probabilities)
        """
        print("Detecting market regimes using Gaussian Mixture Models...")
        
        # Prepare features if not already done
        if self.regime_features is None:
            self.prepare_features()
        
        # Validate that we have data to work with
        if self.regime_features is None or self.regime_features.empty:
            print("Error: No valid features for regime detection")
            # Fall back to time-based regimes
            self._create_time_based_regimes()
            return self.regimes, self.regime_mapping, self.regime_probs
        
        features = self.regime_features
        
        try:
            # 1. Preprocess features
            features_cleaned = features.copy()
            
            # 2. Standardize features using robust scaling to handle outliers
            try:
                scaler = RobustScaler()
                features_scaled = scaler.fit_transform(features_cleaned)
                
                # Check for invalid values after scaling
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                    print("Warning: Scaling produced invalid values. Using simple standardization.")
                    # Use simple Z-score standardization with clipping for outliers
                    means = np.nanmean(features_cleaned.values, axis=0)
                    stds = np.nanstd(features_cleaned.values, axis=0)
                    stds[stds < 1e-8] = 1.0  # Avoid division by zero
                    features_scaled = (features_cleaned.values - means) / stds
                    # Clip extreme values
                    features_scaled = np.clip(features_scaled, -5, 5)
            except Exception as e:
                print(f"Warning: Error in standardization: {str(e)}")
                # Fall back to simple robust centering
                features_scaled = features_cleaned.values.copy()
                for i in range(features_scaled.shape[1]):
                    median = np.median(features_scaled[:, i])
                    mad = np.median(np.abs(features_scaled[:, i] - median))
                    if mad > 1e-8:  # Avoid division by zero
                        features_scaled[:, i] = (features_scaled[:, i] - median) / mad
                    else:
                        features_scaled[:, i] = features_scaled[:, i] - median
                # Clip extreme values
                features_scaled = np.clip(features_scaled, -5, 5)
            
            # 3. Apply dimensionality reduction
            try:
                # Use fewer components to avoid numerical issues
                n_features = features_scaled.shape[1]
                n_samples = features_scaled.shape[0]
                
                # Choose appropriate number of components
                n_components = min(5, n_features // 3, n_samples // 20)
                n_components = max(2, n_components)  # At least 2 components
                
                print(f"Reducing to {n_components} principal components")
                
                # Use PCA with SVD solver which is more numerically stable
                pca = PCA(n_components=n_components, svd_solver='full')
                features_reduced = pca.fit_transform(features_scaled)
                
                # Check for numerical issues after PCA
                if (np.isnan(features_reduced).any() or np.isinf(features_reduced).any() or
                   np.abs(features_reduced).max() > 1e6):
                    print("Warning: PCA produced extreme values. Using a more stable approach.")
                    # Fall back to a more stable approach: standardize and truncate
                    features_reduced = stats.zscore(features_scaled, axis=0)
                    features_reduced = np.clip(features_reduced, -3, 3)
                    # Take just the first few columns as "components"
                    features_reduced = features_reduced[:, :n_components]
            except Exception as e:
                print(f"Warning: Error in dimensionality reduction: {str(e)}")
                # Fall back to using a subset of standardized features
                features_reduced = np.clip(features_scaled, -3, 3)
                if features_reduced.shape[1] > 5:
                    # Use the columns with highest variance if we have many
                    vars = np.nanvar(features_reduced, axis=0)
                    top_idx = np.argsort(-vars)[:5]  # Indices of top 5 variance columns
                    features_reduced = features_reduced[:, top_idx]
            
            # 4. Apply Gaussian Mixture Model
            try:
                # Use fewer regimes for stability
                n_regimes = min(self.n_regimes, 5)
                n_regimes = max(n_regimes, 2)  # At least 2 regimes
                
                print(f"Clustering with {n_regimes} regimes using Gaussian Mixture Model")
                
                # Configure GMM with parameters for numerical stability
                gmm = GaussianMixture(
                    n_components=n_regimes,
                    covariance_type='full',  # Use full covariance matrix
                    random_state=42,
                    reg_covar=1e-4,  # Add regularization
                    n_init=10,       # Try multiple initializations
                    max_iter=200     # Allow more iterations to converge
                )
                
                # Fit and predict
                gmm.fit(features_reduced)
                regime_labels = gmm.predict(features_reduced)
                regime_probs = gmm.predict_proba(features_reduced)
                
                # Create Series with labels
                regimes = pd.Series(regime_labels, index=features.index)
                
                # Create DataFrame with probabilities
                regime_probs_df = pd.DataFrame(
                    regime_probs,
                    index=features.index,
                    columns=[f'Regime_{i}' for i in range(n_regimes)]
                )
                
                # Calculate transition matrix
                transition_matrix = np.zeros((n_regimes, n_regimes))
                prev_regime = regimes.iloc[0]
                
                for curr_regime in regimes.iloc[1:]:
                    transition_matrix[prev_regime, curr_regime] += 1
                    prev_regime = curr_regime
                
                # Normalize to get probabilities
                for i in range(n_regimes):
                    row_sum = transition_matrix[i, :].sum()
                    if row_sum > 0:
                        transition_matrix[i, :] /= row_sum
                
                # Store results
                self.regimes = regimes
                self.regime_probs = regime_probs_df
                self.transition_matrix = transition_matrix
                
                # Name the regimes
                self._name_regimes()
                
                print(f"Successfully detected {n_regimes} regimes")
                return self.regimes, self.regime_mapping, self.regime_probs
                
            except Exception as e:
                print(f"Warning: Error in GMM clustering: {str(e)}")
                # Fall back to K-means which is more robust
                try:
                    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=20)
                    regime_labels = kmeans.fit_predict(features_reduced)
                    regimes = pd.Series(regime_labels, index=features.index)
                    
                    # Create dummy probabilities (just 0/1 for K-means)
                    dummy_probs = np.zeros((len(regimes), n_regimes))
                    for i, label in enumerate(regime_labels):
                        dummy_probs[i, label] = 1.0
                    
                    regime_probs_df = pd.DataFrame(
                        dummy_probs,
                        index=features.index,
                        columns=[f'Regime_{i}' for i in range(n_regimes)]
                    )
                    
                    # Calculate transition matrix
                    transition_matrix = np.zeros((n_regimes, n_regimes))
                    prev_regime = regimes.iloc[0]
                    
                    for curr_regime in regimes.iloc[1:]:
                        transition_matrix[prev_regime, curr_regime] += 1
                        prev_regime = curr_regime
                    
                    # Normalize to get probabilities
                    for i in range(n_regimes):
                        row_sum = transition_matrix[i, :].sum()
                        if row_sum > 0:
                            transition_matrix[i, :] /= row_sum
                    
                    # Store results
                    self.regimes = regimes
                    self.regime_probs = regime_probs_df
                    self.transition_matrix = transition_matrix
                    
                    # Name the regimes
                    self._name_regimes()
                    
                    print(f"Successfully detected {n_regimes} regimes using K-means fallback")
                    return self.regimes, self.regime_mapping, self.regime_probs
                    
                except Exception as e2:
                    print(f"Error in K-means fallback: {str(e2)}")
                    # Fall back to time-based regimes as last resort
                    self._create_time_based_regimes()
                    return self.regimes, self.regime_mapping, self.regime_probs
        
        except Exception as e:
            print(f"Error in regime detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to time-based regimes
            self._create_time_based_regimes()
            return self.regimes, self.regime_mapping, self.regime_probs
    
    def _create_time_based_regimes(self):
        """Create time-based regimes as a fallback method."""
        print("Creating time-based regimes as fallback...")
        
        if self.returns is None or self.returns.empty:
            dates = pd.date_range(start='2018-01-01', periods=100)
        else:
            dates = self.returns.index
        
        n_regimes = min(self.n_regimes, 5)
        
        # Create regimes based on equal time periods
        regimes = pd.Series(0, index=dates)
        segment_size = len(dates) // n_regimes
        
        for i in range(1, n_regimes):
            start_idx = i * segment_size
            regimes.iloc[start_idx:] = i
        
        # Create dummy probabilities
        regime_probs = pd.DataFrame(0, index=dates, columns=[f'Regime_{i}' for i in range(n_regimes)])
        for i, regime in enumerate(regimes):
            regime_probs.iloc[i, regime] = 1.0
        
        # Create transition matrix (default to 0.8 staying in same regime, 0.2 to next)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            transition_matrix[i, i] = 0.8  # 80% chance of staying in same regime
            next_regime = (i + 1) % n_regimes
            transition_matrix[i, next_regime] = 0.2  # 20% chance of moving to next regime
        
        # Store results
        self.regimes = regimes
        self.regime_probs = regime_probs
        self.transition_matrix = transition_matrix
        
        # Create regime names
        regime_names = ["Risk-On", "Trade Tension", "Flight to Safety", "Inflation", "Crisis"]
        regime_names = regime_names[:n_regimes]  # Take only as many as we need
        
        self.regime_mapping = {i: name for i, name in enumerate(regime_names)}
        self.named_regimes = self.regimes.map(self.regime_mapping)
        
        print(f"Created {n_regimes} time-based regimes")
    
    def _name_regimes(self):
        """Name the detected regimes based on their characteristics."""
        if self.regimes is None or self.returns is None:
            # Default naming if we don't have data
            self.regime_mapping = {i: f"Regime_{i}" for i in range(self.regimes.nunique())}
            self.named_regimes = self.regimes.map(self.regime_mapping)
            return
        
        try:
            # Calculate characteristics of each regime
            returns = self.returns
            
            # Analysis across different asset classes
            regime_profiles = {}
            
            for regime in sorted(self.regimes.unique()):
                # Get returns data for this regime
                regime_dates = self.regimes[self.regimes == regime].index
                regime_returns = returns.loc[regime_dates]
                
                # Calculate statistics per asset class
                profile = {}
                
                for asset_class, assets in self.asset_classes.items():
                    valid_assets = [a for a in assets if a in returns.columns]
                    if not valid_assets:
                        continue
                    
                    # Calculate average return and volatility
                    class_returns = regime_returns[valid_assets]
                    profile[f"{asset_class}_return"] = class_returns.mean().mean()
                    profile[f"{asset_class}_vol"] = class_returns.std().mean()
                    
                    # Calculate correlation with other asset classes
                    for other_class, other_assets in self.asset_classes.items():
                        if asset_class != other_class:
                            other_valid = [a for a in other_assets if a in returns.columns]
                            if other_valid:
                                class_avg = class_returns.mean(axis=1)
                                other_avg = regime_returns[other_valid].mean(axis=1)
                                profile[f"{asset_class}_{other_class}_corr"] = class_avg.corr(other_avg)
                
                # Add economic data if available
                if self.economic_data is not None:
                    common_dates = regime_dates.intersection(self.economic_data.index)
                    if len(common_dates) > 0:
                        econ_data = self.economic_data.loc[common_dates]
                        
                        for col in econ_data.columns:
                            if not econ_data[col].empty:
                                profile[col] = econ_data[col].mean()
                
                # Store profile
                regime_profiles[regime] = profile
            
            # Name regimes based on profiles
            regime_names = {}
            
            # Common regime patterns to detect
            for regime, profile in regime_profiles.items():
                # Extract key statistics
                equity_return = profile.get("Equities_return", 0)
                equity_vol = profile.get("Equities_vol", 0)
                bond_return = profile.get("Bonds_return", 0)
                commodity_return = profile.get("Commodities_return", 0)
                
                # Get economic indicators if available
                tariff_impact = profile.get("Tariff_Impact", 0)
                uncertainty = profile.get("EPU_Index", 0)
                
                # Determine regime type
                if tariff_impact > 70 or uncertainty > 70:
                    if equity_return < 0:
                        regime_names[regime] = "Trade Tension"
                    else:
                        regime_names[regime] = "High Uncertainty"
                elif equity_return > 0 and bond_return > 0:
                    if equity_return > 0.01:  # Strong equity returns
                        regime_names[regime] = "Risk-On"
                    else:
                        regime_names[regime] = "Steady State"
                elif equity_return < 0 and bond_return > 0:
                    regime_names[regime] = "Flight to Safety"
                elif equity_return < 0 and bond_return < 0:
                    if equity_vol > 0.015:  # High volatility
                        regime_names[regime] = "Crisis"
                    else:
                        regime_names[regime] = "Inflation"
                elif commodity_return > max(equity_return, bond_return):
                    regime_names[regime] = "Commodity Bull"
                else:
                    regime_names[regime] = f"Regime_{regime}"
            
            # Store regime mapping
            self.regime_mapping = regime_names
            self.named_regimes = self.regimes.map(self.regime_mapping)
            
            print("Named regimes based on characteristics:")
            for regime, name in regime_names.items():
                print(f"  Regime {regime}: {name}")
            
        except Exception as e:
            print(f"Error naming regimes: {str(e)}")
            # Default naming as fallback
            self.regime_mapping = {i: f"Regime_{i}" for i in range(self.regimes.nunique())}
            self.named_regimes = self.regimes.map(self.regime_mapping)
    
    def get_current_regime(self, with_probability=False):
        """
        Get the current market regime with probability if requested.
        
        Parameters:
        -----------
        with_probability : bool, optional
            Whether to return the probability along with the regime (default: False)
        
        Returns:
        --------
        str or tuple
            Current regime name or (name, probability)
        """
        if self.named_regimes is None:
            self.detect_regimes()
        
        # Get the most recent regime
        current_regime = self.named_regimes.iloc[-1]
        
        if with_probability and self.regime_probs is not None:
            current_idx = self.regimes.iloc[-1]
            current_prob = self.regime_probs.iloc[-1, current_idx]
            return current_regime, current_prob
        
        return current_regime
    
    def predict_next_regime(self):
        """
        Predict the next regime based on transition probabilities.
        
        Returns:
        --------
        tuple
            (next_regime_name, probability)
        """
        if self.transition_matrix is None or self.regimes is None:
            self.detect_regimes()
        
        # Get current regime
        current_regime = self.regimes.iloc[-1]
        
        # Get transition probabilities from current regime
        transition_probs = self.transition_matrix[current_regime, :]
        
        # Find most likely next regime
        next_regime = np.argmax(transition_probs)
        next_prob = transition_probs[next_regime]
        
        # Map to name
        next_name = self.regime_mapping.get(next_regime, f"Regime_{next_regime}")
        
        return next_name, next_prob
    
    def get_regime_probability(self, regime_name):
        """
        Get the probability of a specific regime based on current data.
        
        Parameters:
        -----------
        regime_name : str
            Name of the regime to check
        
        Returns:
        --------
        float
            Probability of the specified regime
        """
        if self.regime_probs is None or self.regime_mapping is None:
            self.detect_regimes()
        
        # Get regime index from name
        regime_indices = [k for k, v in self.regime_mapping.items() if v == regime_name]
        
        if not regime_indices:
            return 0.0
        
        # Sum probabilities for all matching indices (if same name assigned to multiple indices)
        total_prob = 0.0
        for idx in regime_indices:
            col_name = f'Regime_{idx}'
            if col_name in self.regime_probs.columns:
                total_prob += self.regime_probs.iloc[-1][col_name]
        
        return total_prob
    
    def visualize_regimes(self, show_probabilities=True):
        """
        Visualize the detected regimes with probabilities and asset class performance.
        
        Parameters:
        -----------
        show_probabilities : bool, optional
            Whether to show regime probabilities (default: True)
        
        Returns:
        --------
        None
        """
        if self.regimes is None or self.named_regimes is None:
            print("No regimes detected. Run detect_regimes() first.")
            return
        
        # Set up figure
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Regime transitions over time
        plt.subplot(3, 1, 1)
        
        try:
            # Plot regime transitions with background color
            n_regimes = len(self.regime_mapping)
            colors = plt.cm.tab10(np.linspace(0, 1, n_regimes))
            
            # Get unique regime spans
            regime_spans = []
            current_regime = self.named_regimes.iloc[0]
            start_date = self.named_regimes.index[0]
            
            for i, (date, regime) in enumerate(self.named_regimes.items()):
                if regime != current_regime or i == len(self.named_regimes) - 1:
                    # Store completed span
                    end_date = date
                    regime_spans.append((start_date, end_date, current_regime))
                    
                    # Start new span
                    start_date = date
                    current_regime = regime
            
            # Plot spans
            for i, (start, end, regime) in enumerate(regime_spans):
                # Get color index
                regime_idx = [k for k, v in self.regime_mapping.items() if v == regime][0]
                color = colors[regime_idx % len(colors)]
                
                plt.axvspan(start, end, color=color, alpha=0.3)
                
                # Add text label in the middle of the span
                mid_date = start + (end - start) / 2
                plt.text(mid_date, 0.5, regime, fontsize=9, 
                        ha='center', va='center', color='black')
            
            plt.yticks([])
            plt.title("Market Regime Transitions Over Time")
            plt.xticks(rotation=45)
        except Exception as e:
            print(f"Error plotting regime transitions: {str(e)}")
            plt.text(0.5, 0.5, "Could not plot regime transitions", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 2: Regime probabilities if requested
        if show_probabilities and self.regime_probs is not None:
            plt.subplot(3, 1, 2)
            
            try:
                # Plot probabilities
                plot_data = self.regime_probs.copy()
                
                # Rename columns with regime names
                new_columns = {}
                for col in plot_data.columns:
                    # Extract regime index from column name
                    if col.startswith('Regime_'):
                        try:
                            regime_idx = int(col.split('_')[1])
                            regime_name = self.regime_mapping.get(regime_idx, col)
                            new_columns[col] = regime_name
                        except ValueError:
                            new_columns[col] = col
                
                plot_data = plot_data.rename(columns=new_columns)
                
                # Plot stacked area
                plot_data.plot(kind='area', stacked=True, alpha=0.7, ax=plt.gca())
                
                plt.title("Regime Probabilities Over Time")
                plt.ylabel("Probability")
                plt.ylim(0, 1)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xticks(rotation=45)
            except Exception as e:
                print(f"Error plotting regime probabilities: {str(e)}")
                plt.text(0.5, 0.5, "Could not plot regime probabilities", 
                        ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 3: Asset class returns by regime
        plt.subplot(3, 1, 3)
        
        try:
            # Calculate returns for each asset class in each regime
            if self.returns is not None and not self.returns.empty:
                # Get common dates between returns and regimes
                common_dates = self.regimes.index.intersection(self.returns.index)
                
                if len(common_dates) > 0:
                    # Initialize results
                    asset_class_returns = {}
                    
                    for regime_name in self.named_regimes.unique():
                        # Get dates for this regime
                        regime_dates = common_dates[self.named_regimes.loc[common_dates] == regime_name]
                        
                        if len(regime_dates) > 0:
                            # Calculate returns for each asset class
                            class_returns = []
                            
                            for asset_class, assets in self.asset_classes.items():
                                valid_assets = [a for a in assets if a in self.returns.columns]
                                
                                if valid_assets:
                                    # Calculate average return for this asset class in this regime
                                    avg_return = self.returns.loc[regime_dates, valid_assets].mean().mean()
                                    class_returns.append((asset_class, avg_return))
                            
                            asset_class_returns[regime_name] = class_returns
                    
                    # Convert to DataFrame for plotting
                    plot_data = pd.DataFrame({
                        regime: {asset_class: ret for asset_class, ret in returns}
                        for regime, returns in asset_class_returns.items()
                    })
                    
                    # Plot as bar chart
                    plot_data.plot(kind='bar', ax=plt.gca())
                    plt.title("Average Returns by Asset Class and Regime")
                    plt.ylabel("Average Daily Return")
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.xticks(rotation=45)
                else:
                    plt.text(0.5, 0.5, "No common dates between returns and regimes", 
                             ha='center', va='center', transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, "No returns data available", 
                         ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            print(f"Error plotting asset class returns: {str(e)}")
            plt.text(0.5, 0.5, "Could not plot asset class returns", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 4: Transition probability matrix
        plt.figure(figsize=(10, 8))
        
        try:
            if self.transition_matrix is not None:
                # Create DataFrame with regime names
                regime_names = [self.regime_mapping.get(i, f"Regime_{i}") 
                               for i in range(len(self.transition_matrix))]
                
                transition_df = pd.DataFrame(
                    self.transition_matrix,
                    index=regime_names,
                    columns=regime_names
                )
                
                # Plot heatmap
                sns.heatmap(transition_df, annot=True, cmap='Blues', vmin=0, vmax=1,
                           linewidths=0.5, cbar_kws={'label': 'Transition Probability'})
                
                plt.title("Regime Transition Probability Matrix")
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, "No transition matrix available", 
                         ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            print(f"Error plotting transition matrix: {str(e)}")
            plt.text(0.5, 0.5, "Could not plot transition matrix", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def save_regimes(self, filepath='data/enhanced_regimes.csv'):
        """
        Save detected regimes to a CSV file with probabilities and transition matrix.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the regimes (default: 'data/enhanced_regimes.csv')
        
        Returns:
        --------
        None
        """
        if self.named_regimes is not None:
            # Create base directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save regimes
            regimes_df = pd.DataFrame({
                'numeric_regime': self.regimes,
                'named_regime': self.named_regimes
            })
            
            # Add regime probabilities if available
            if self.regime_probs is not None:
                for col in self.regime_probs.columns:
                    regimes_df[col] = self.regime_probs[col]
            
            regimes_df.to_csv(filepath)
            
            # Save transition matrix
            if self.transition_matrix is not None:
                transition_df = pd.DataFrame(
                    self.transition_matrix,
                    index=[f"From_Regime_{i}" for i in range(len(self.transition_matrix))],
                    columns=[f"To_Regime_{i}" for i in range(len(self.transition_matrix))]
                )
                transition_df.to_csv(filepath.replace('.csv', '_transitions.csv'))
            
            # Save regime mapping
            if self.regime_mapping is not None:
                mapping_df = pd.DataFrame({
                    'regime_id': list(self.regime_mapping.keys()),
                    'regime_name': list(self.regime_mapping.values())
                })
                mapping_df.to_csv(filepath.replace('.csv', '_mapping.csv'), index=False)
            
            print(f"Regimes and related data saved to {filepath}")
        else:
            print("No regimes to save. Run detect_regimes() first.")
    
    def load_regimes(self, filepath='data/enhanced_regimes.csv'):
        """
        Load detected regimes from a CSV file with probabilities and transition matrix.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to load the regimes from (default: 'data/enhanced_regimes.csv')
        
        Returns:
        --------
        tuple
            (regimes, named_regimes, regime_probs)
        """
        try:
            # Load regimes
            regimes_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            self.regimes = regimes_df['numeric_regime']
            self.named_regimes = regimes_df['named_regime']
            
            # Extract regime probabilities if available
            prob_columns = [col for col in regimes_df.columns if col.startswith('Regime_')]
            if prob_columns:
                self.regime_probs = regimes_df[prob_columns]
            
            # Load transition matrix if available
            transition_path = filepath.replace('.csv', '_transitions.csv')
            if os.path.exists(transition_path):
                transition_df = pd.read_csv(transition_path, index_col=0)
                self.transition_matrix = transition_df.values
            
            # Load regime mapping if available
            mapping_path = filepath.replace('.csv', '_mapping.csv')
            if os.path.exists(mapping_path):
                mapping_df = pd.read_csv(mapping_path)
                self.regime_mapping = dict(zip(mapping_df['regime_id'], mapping_df['regime_name']))
            else:
                # Reconstruct regime mapping from named_regimes
                self.regime_mapping = {}
                for regime, name in zip(self.regimes, self.named_regimes):
                    self.regime_mapping[regime] = name
            
            print(f"Regimes and related data loaded from {filepath}")
            return self.regimes, self.named_regimes, self.regime_probs
        except FileNotFoundError:
            print(f"Regime file {filepath} not found.")
            return None, None, None
        except Exception as e:
            print(f"Error loading regimes: {str(e)}")
            return None, None, None