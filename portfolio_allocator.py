"""
Enhanced portfolio allocation module that ensures proper diversification
across all market regimes with minimum exposure requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BalancedPortfolioAllocator:
    """
    Class for dynamic portfolio allocation with balanced exposures and minimum
    diversification constraints across different market regimes.
    """
    
    def __init__(self, trend_directions, trend_strengths, regime=None, regime_probs=None, 
                 max_position_size=0.20, min_position_count=3, leverage=1.0, 
                 min_long_exposure=0.30, min_short_exposure=0.20):
        """
        Initialize the BalancedPortfolioAllocator.
        
        Parameters:
        -----------
        trend_directions : pd.Series
            Direction predictions for each asset (1 for up, -1 for down)
        trend_strengths : pd.Series
            Strength of trend predictions for each asset
        regime : str, optional
            Current market regime (default: None)
        regime_probs : pd.Series, optional
            Probabilities of different regimes (default: None)
        max_position_size : float, optional
            Maximum weight for any single position (default: 0.20)
        min_position_count : int, optional
            Minimum number of positions in each direction (default: 3)
        leverage : float, optional
            Total leverage (default: 1.0)
        min_long_exposure : float, optional
            Minimum long exposure in portfolio (default: 0.30)
        min_short_exposure : float, optional
            Minimum short exposure in portfolio (default: 0.20)
        """
        # Store parameters
        self.trend_directions = trend_directions
        self.trend_strengths = trend_strengths
        self.regime = regime
        self.regime_probs = regime_probs
        self.max_position_size = max_position_size
        self.min_position_count = min_position_count
        self.leverage = leverage
        self.min_long_exposure = min_long_exposure
        self.min_short_exposure = min_short_exposure
        
        # Initialize allocation
        self.weights = None
        self.allocation_details = {}
        
        # Define asset class groupings (can be overridden later)
        self._setup_asset_classes()
    
    def _setup_asset_classes(self):
        """Setup asset class groupings based on columns."""
        # Set up default asset classes based on names
        columns = self.trend_directions.index
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
    
    def set_asset_classes(self, asset_classes):
        """
        Override default asset class groupings.
        
        Parameters:
        -----------
        asset_classes : dict
            Dictionary mapping asset class names to lists of assets
        """
        self.asset_classes = asset_classes
    
    def determine_regime_allocations(self):
        """
        Determine asset class allocations based on the current regime.
        
        Returns:
        --------
        dict
            Asset class allocation weights
        """
        if self.regime is None:
            # Default balanced allocation
            return {
                'Commodities': 0.25,
                'Currencies': 0.25,
                'Bonds': 0.25,
                'Equities': 0.25,
                'Others': 0.00  # Will be allocated if 'Others' category exists
            }
        
        # Different allocation approach based on regime
        if "Trade Tension" in self.regime:
            # In Trade Tension regime, focus on defensive assets but remain diversified
            return {
                'Commodities': 0.35,  # Domestic commodities may benefit
                'Currencies': 0.20,   # USD often strengthens
                'Bonds': 0.30,        # Flight to safety
                'Equities': 0.15,     # Reduced but not eliminated equity exposure
                'Others': 0.00
            }
        elif "High Uncertainty" in self.regime or "Flight to Safety" in self.regime:
            # In High Uncertainty regime, reduce risk but maintain diversification
            return {
                'Commodities': 0.20,
                'Currencies': 0.20,
                'Bonds': 0.45,  # Heavy but not exclusive bond focus
                'Equities': 0.15,
                'Others': 0.00
            }
        elif "Risk-On" in self.regime or "Steady State" in self.regime:
            # In Risk-On regime, focus more on equities and commodities
            return {
                'Commodities': 0.25,
                'Currencies': 0.15,
                'Bonds': 0.20,
                'Equities': 0.40,  # Highest equity allocation
                'Others': 0.00
            }
        elif "Crisis" in self.regime:
            # In Crisis, more balanced with focus on bonds and defensive assets
            return {
                'Commodities': 0.25,
                'Currencies': 0.25,
                'Bonds': 0.35,
                'Equities': 0.15,
                'Others': 0.00
            }
        elif "Inflation" in self.regime:
            # In Inflation, commodities tend to do well, bonds poorly
            return {
                'Commodities': 0.40,
                'Currencies': 0.20,
                'Bonds': 0.15,
                'Equities': 0.25,
                'Others': 0.00
            }
        elif "Commodity" in self.regime:
            # In Commodity Bull, heavy focus on commodities
            return {
                'Commodities': 0.50,
                'Currencies': 0.15,
                'Bonds': 0.15,
                'Equities': 0.20,
                'Others': 0.00
            }
        else:
            # Default balanced allocation for unknown regimes
            return {
                'Commodities': 0.25,
                'Currencies': 0.25,
                'Bonds': 0.25,
                'Equities': 0.25,
                'Others': 0.00
            }
    
    def blend_regime_allocations(self):
        """
        Blend multiple regime allocations based on regime probabilities.
        
        Returns:
        --------
        dict
            Blended asset class allocation weights
        """
        # If no regime probabilities, just use the current regime
        if self.regime_probs is None or len(self.regime_probs) == 0:
            return self.determine_regime_allocations()
        
        # Initialize blended allocation
        blended_allocation = {
            'Commodities': 0.0,
            'Currencies': 0.0,
            'Bonds': 0.0,
            'Equities': 0.0,
            'Others': 0.0
        }
        
        # Get allocations for each regime and blend based on probabilities
        for regime_name, probability in self.regime_probs.items():
            # Temporarily set the regime to get its allocation
            original_regime = self.regime
            self.regime = regime_name
            
            # Get allocation for this regime
            regime_allocation = self.determine_regime_allocations()
            
            # Blend into result based on probability
            for asset_class, weight in regime_allocation.items():
                blended_allocation[asset_class] += weight * probability
            
            # Restore original regime
            self.regime = original_regime
        
        return blended_allocation
    
    def allocate_portfolio(self, ensure_min_exposures=True):
        """
        Dynamic portfolio allocation that ensures both long and short exposures
        across different asset classes with minimum diversification constraints.
        
        Parameters:
        -----------
        ensure_min_exposures : bool, optional
            Whether to enforce minimum exposure constraints (default: True)
        
        Returns:
        --------
        pd.Series
            Asset allocation weights
        """
        print("Allocating portfolio with balanced exposures...")
        
        # Initialize with zeros
        weights = pd.Series(0.0, index=self.trend_directions.index)
        
        # Determine asset class allocations based on regime(s)
        if self.regime_probs is not None and len(self.regime_probs) > 0:
            allocation_weights = self.blend_regime_allocations()
            print("Using blended allocations from multiple regimes")
        else:
            allocation_weights = self.determine_regime_allocations()
        
        # Display regime and allocations
        if self.regime:
            print(f"Current market regime: {self.regime}")
        print("Asset class allocations:")
        for asset_class, weight in allocation_weights.items():
            if weight > 0:
                print(f"  {asset_class}: {weight:.2%}")
        
        # Store allocation details for later analysis
        self.allocation_details = {
            'regime': self.regime,
            'asset_class_allocations': allocation_weights.copy(),
            'asset_allocations': {},
            'reasons': {}
        }
        
        # For each asset class, ensure both long and short positions
        for asset_class, class_weight in allocation_weights.items():
            if class_weight == 0:
                continue
                
            assets = self.asset_classes.get(asset_class, [])
            if not assets:
                continue
            
            # Get signals for this class
            valid_assets = [a for a in assets if a in self.trend_strengths.index]
            if not valid_assets:
                continue
                
            # Calculate signal strength and direction
            strengths = self.trend_strengths[valid_assets]
            directions = self.trend_directions[valid_assets].copy()
            
            # Special case: if we're in an uncertain regime, increase short bias
            short_allocation = 0.3  # Default 30% allocation to shorts
            if self.regime and ('Tension' in self.regime or 'Uncertainty' in self.regime or 'Crisis' in self.regime):
                short_allocation = 0.4  # 40% to shorts in uncertain regimes
            
            # Make sure we have enough assets for minimum position count
            if len(valid_assets) < self.min_position_count * 2:
                # Not enough assets for minimum count in both directions
                # Just split them as evenly as possible
                split_point = len(valid_assets) // 2
                
                # Sort by strength regardless of direction
                sorted_assets = sorted(valid_assets, key=lambda a: strengths[a])
                
                # Force bottom half to be shorts
                for asset in sorted_assets[:split_point]:
                    directions[asset] = -1
                
                # Force top half to be longs
                for asset in sorted_assets[split_point:]:
                    directions[asset] = 1
            else:
                # Ensure minimum number of positions in each direction
                
                # First, find the naturally long and short assets
                natural_longs = [a for a in valid_assets if directions[a] > 0]
                natural_shorts = [a for a in valid_assets if directions[a] < 0]
                
                # If we don't have enough shorts, force some long positions to be shorts
                if len(natural_shorts) < self.min_position_count:
                    # Sort longs by strength (ascending) and convert weakest to shorts
                    if natural_longs:
                        longs_by_strength = sorted(natural_longs, key=lambda a: strengths[a])
                        shorts_needed = self.min_position_count - len(natural_shorts)
                        shorts_to_add = min(shorts_needed, len(longs_by_strength) - self.min_position_count)
                        
                        if shorts_to_add > 0:
                            for asset in longs_by_strength[:shorts_to_add]:
                                directions[asset] = -1
                                self.allocation_details['reasons'][asset] = "Converted to short to meet minimum short count"
                
                # If we don't have enough longs, force some short positions to be longs
                if len(natural_longs) < self.min_position_count:
                    # Sort shorts by strength (ascending) and convert weakest to longs
                    if natural_shorts:
                        shorts_by_strength = sorted(natural_shorts, key=lambda a: strengths[a])
                        longs_needed = self.min_position_count - len(natural_longs)
                        longs_to_add = min(longs_needed, len(shorts_by_strength) - self.min_position_count)
                        
                        if longs_to_add > 0:
                            for asset in shorts_by_strength[:longs_to_add]:
                                directions[asset] = 1
                                self.allocation_details['reasons'][asset] = "Converted to long to meet minimum long count"
            
            # Now proceed with allocation
            long_assets = [a for a in valid_assets if directions[a] > 0]
            short_assets = [a for a in valid_assets if directions[a] < 0]
            
            # Default allocation: 70% to longs, 30% to shorts (or as adjusted above)
            long_allocation = 1.0 - short_allocation
            
            # Special case for bonds in flight to safety
            if asset_class == 'Bonds' and self.regime and 'Safety' in self.regime:
                # More heavily weight long bonds in flight to safety
                long_allocation = 0.8
                short_allocation = 0.2
            
            # Special case for equities in crisis
            if asset_class == 'Equities' and self.regime and 'Crisis' in self.regime:
                # More heavily weight short equities in crisis
                long_allocation = 0.4
                short_allocation = 0.6
            
            # Allocate for long positions
            if long_assets:
                # Extract and renormalize strengths for long assets
                long_strengths = strengths[long_assets]
                if long_strengths.sum() > 0:
                    norm_long_strengths = long_strengths / long_strengths.sum()
                else:
                    norm_long_strengths = pd.Series(1.0/len(long_assets), index=long_assets)
                
                for asset in long_assets:
                    weights[asset] = class_weight * long_allocation * norm_long_strengths[asset]
                    self.allocation_details['asset_allocations'][asset] = {
                        'direction': 'long',
                        'raw_weight': weights[asset],
                        'asset_class': asset_class,
                        'strength': strengths[asset]
                    }
            
            # Allocate for short positions
            if short_assets:
                # Extract and renormalize strengths for short assets
                short_strengths = strengths[short_assets]
                if short_strengths.sum() > 0:
                    norm_short_strengths = short_strengths / short_strengths.sum()
                else:
                    norm_short_strengths = pd.Series(1.0/len(short_assets), index=short_assets)
                
                for asset in short_assets:
                    # Note the negative sign for short positions
                    weights[asset] = -class_weight * short_allocation * norm_short_strengths[asset]
                    self.allocation_details['asset_allocations'][asset] = {
                        'direction': 'short',
                        'raw_weight': weights[asset],
                        'asset_class': asset_class,
                        'strength': strengths[asset]
                    }
            
            # If we have no short assets, force at least one
            if not short_assets and long_assets and len(long_assets) > self.min_position_count:
                # Take the weakest long asset and make it a short
                weakest_asset = long_strengths.idxmin()
                # Remove from long allocation
                weights[weakest_asset] = 0
                # Add as short
                weights[weakest_asset] = -class_weight * short_allocation
                self.allocation_details['reasons'][weakest_asset] = "Converted to short as asset class had no shorts"
                self.allocation_details['asset_allocations'][weakest_asset] = {
                    'direction': 'short',
                    'raw_weight': weights[weakest_asset],
                    'asset_class': asset_class,
                    'strength': strengths[weakest_asset]
                }
        
        # Apply leverage
        weights = weights * self.leverage
        
        # Apply risk management constraints
        # No single asset can be more than max_position_size of portfolio
        weights = weights.clip(lower=-self.max_position_size, upper=self.max_position_size)
        
        # Calculate current exposures
        long_exposure = weights[weights > 0].sum()
        short_exposure = abs(weights[weights < 0].sum())
        total_exposure = long_exposure + short_exposure
        
        # Ensure minimum exposures if requested and if we have non-zero weights
        if ensure_min_exposures and total_exposure > 0:
            # Check if we need to adjust
            adjustment_needed = False
            
            if long_exposure < self.min_long_exposure * self.leverage:
                adjustment_needed = True
                print(f"Long exposure {long_exposure:.2%} below minimum {(self.min_long_exposure * self.leverage):.2%}")
                
            if short_exposure < self.min_short_exposure * self.leverage:
                adjustment_needed = True
                print(f"Short exposure {short_exposure:.2%} below minimum {(self.min_short_exposure * self.leverage):.2%}")
            
            if adjustment_needed:
                print("Adjusting allocations to meet minimum exposure requirements")
                
                # Calculate target exposures that maintain current ratio but meet minimums
                min_long = self.min_long_exposure * self.leverage
                min_short = self.min_short_exposure * self.leverage
                
                # If both are below minimum, scale up both sides
                if long_exposure < min_long and short_exposure < min_short:
                    scale_long = min_long / max(long_exposure, 1e-6)
                    scale_short = min_short / max(short_exposure, 1e-6)
                    
                    # Scale all long positions
                    for asset in weights.index:
                        if weights[asset] > 0:
                            weights[asset] *= scale_long
                        elif weights[asset] < 0:
                            weights[asset] *= scale_short
                
                # If just one side is below minimum, adjust to maintain total leverage
                elif long_exposure < min_long:
                    # Increase longs to minimum
                    old_long = long_exposure
                    long_exposure = min_long
                    
                    # Scale down shorts to maintain total leverage
                    target_short = min(short_exposure, self.leverage - long_exposure)
                    short_scale = target_short / max(short_exposure, 1e-6)
                    
                    # Apply scaling
                    for asset in weights.index:
                        if weights[asset] > 0:
                            weights[asset] *= (long_exposure / max(old_long, 1e-6))
                        elif weights[asset] < 0:
                            weights[asset] *= short_scale
                
                elif short_exposure < min_short:
                    # Increase shorts to minimum
                    old_short = short_exposure
                    short_exposure = min_short
                    
                    # Scale down longs to maintain total leverage
                    target_long = min(long_exposure, self.leverage - short_exposure)
                    long_scale = target_long / max(long_exposure, 1e-6)
                    
                    # Apply scaling
                    for asset in weights.index:
                        if weights[asset] < 0:
                            weights[asset] *= (short_exposure / max(old_short, 1e-6))
                        elif weights[asset] > 0:
                            weights[asset] *= long_scale
        
        # Final check: ensure no position exceeds max_position_size after adjustments
        weights = weights.clip(lower=-self.max_position_size, upper=self.max_position_size)
        
        # Normalize to ensure total exposure equals leverage
        total = abs(weights).sum()
        if total > 0:
            weights = weights / total * self.leverage
        
        # Store the weights
        self.weights = weights
        
        # Print long-short exposure
        long_exposure, short_exposure = self.get_long_short_exposure()
        print(f"Long exposure: {long_exposure:.2%}, Short exposure: {short_exposure:.2%}")
        print(f"Total exposure: {(long_exposure + short_exposure):.2%} (leverage: {self.leverage})")
        
        print("Portfolio allocation complete")
        return self.weights
    
    def get_exposure_by_asset_class(self):
        """
        Calculate exposure by asset class.
        
        Returns:
        --------
        tuple
            (gross_exposure, net_exposure) by asset class
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return None, None
        
        gross_exposure = {}
        net_exposure = {}
        
        for asset_class, assets in self.asset_classes.items():
            if assets:
                # Get assets that exist in our weights
                valid_assets = [a for a in assets if a in self.weights.index]
                if valid_assets:
                    # Calculate exposures
                    weights = self.weights[valid_assets]
                    gross_exposure[asset_class] = abs(weights).sum()
                    net_exposure[asset_class] = weights.sum()
        
        return pd.Series(gross_exposure), pd.Series(net_exposure)
    
    def get_long_short_exposure(self):
        """
        Calculate long and short exposure.
        
        Returns:
        --------
        tuple
            (long_exposure, short_exposure)
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return 0.0, 0.0
        
        long_exposure = self.weights[self.weights > 0].sum()
        short_exposure = abs(self.weights[self.weights < 0].sum())
        
        return long_exposure, short_exposure
    
    def get_top_positions(self, n=5):
        """
        Get the top long and short positions.
        
        Parameters:
        -----------
        n : int, optional
            Number of top positions to return (default: 5)
        
        Returns:
        --------
        tuple
            (top_long, top_short)
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return None, None
        
        long_positions = self.weights[self.weights > 0]
        short_positions = self.weights[self.weights < 0]
        
        top_long = long_positions.nlargest(min(n, len(long_positions)))
        top_short = short_positions.nsmallest(min(n, len(short_positions)))
        
        return top_long, top_short
    
    def get_position_count(self):
        """
        Get the count of long and short positions.
        
        Returns:
        --------
        tuple
            (long_count, short_count)
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return 0, 0
        
        long_count = (self.weights > 0).sum()
        short_count = (self.weights < 0).sum()
        
        return long_count, short_count
    
    def get_allocation_summary(self):
        """
        Get a summary of the allocation.
        
        Returns:
        --------
        dict
            Summary of the allocation
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return {}
        
        # Get exposures
        long_exposure, short_exposure = self.get_long_short_exposure()
        gross_exposure_by_class, net_exposure_by_class = self.get_exposure_by_asset_class()
        long_count, short_count = self.get_position_count()
        
        # Create summary
        summary = {
            'regime': self.regime,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'total_exposure': long_exposure + short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'long_count': long_count,
            'short_count': short_count,
            'total_positions': long_count + short_count,
            'gross_exposure_by_class': gross_exposure_by_class.to_dict(),
            'net_exposure_by_class': net_exposure_by_class.to_dict()
        }
        
        return summary
    
    def visualize_allocation(self):
        """
        Visualize the portfolio allocation with enhanced charts.
        
        Returns:
        --------
        None
        """
        if self.weights is None:
            print("No portfolio weights available. Run allocate_portfolio() first.")
            return
        
        # Set up figure
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Bar chart of all positions
        plt.subplot(3, 1, 1)
        sorted_weights = self.weights.sort_values()
        colors = ['red' if w < 0 else 'green' for w in sorted_weights]
        sorted_weights.plot(kind='barh', color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Portfolio Allocation ({self.regime} Regime)' if self.regime else 'Portfolio Allocation')
        plt.xlabel('Weight')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Treemap of positions by asset class
        plt.subplot(3, 2, 3)
        try:
            import squarify
            
            # Get absolute weights by asset class
            class_weights = {}
            for asset_class, assets in self.asset_classes.items():
                valid_assets = [a for a in assets if a in self.weights.index]
                if valid_assets:
                    abs_weight = abs(self.weights[valid_assets]).sum()
                    if abs_weight > 0:
                        class_weights[asset_class] = abs_weight
            
            if class_weights:
                # Sort by weight
                sorted_classes = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
                labels = [f"{cls}: {weight:.1%}" for cls, weight in sorted_classes]
                sizes = [weight for _, weight in sorted_classes]
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(sizes)))
                
                # Draw treemap
                squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7, pad=True)
                plt.axis('off')
                plt.title('Gross Exposure by Asset Class')
            else:
                plt.text(0.5, 0.5, 'No asset class data available', 
                         ha='center', va='center', transform=plt.gca().transAxes)
        except ImportError:
            # Fallback to pie chart if squarify not available
            gross_exposure, _ = self.get_exposure_by_asset_class()
            if not gross_exposure.empty:
                gross_exposure.plot(kind='pie', autopct='%1.1f%%')
                plt.title('Gross Exposure by Asset Class')
                plt.ylabel('')
            else:
                plt.text(0.5, 0.5, 'No asset class data available', 
                         ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 3: Long/Short bar chart
        plt.subplot(3, 2, 4)
        long_exposure, short_exposure = self.get_long_short_exposure()
        
        # Calculate net exposure
        net_exposure = long_exposure - short_exposure
        
        # Create bar chart
        exposure_df = pd.Series({
            'Long': long_exposure, 
            'Short': -short_exposure,  # Negative for visualization
            'Net': net_exposure
        })
        
        colors = ['green', 'red', 'blue']
        exposure_df.plot(kind='bar', color=colors)
        
        plt.title('Portfolio Exposure')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.ylabel('Exposure')
        plt.grid(True, alpha=0.3)
        
        # Add text annotations with percentages
        for i, value in enumerate(exposure_df):
            plt.text(i, value + (0.05 if value > 0 else -0.05), 
                     f"{abs(value):.1%}", ha='center')
        
        # Plot 4: Net exposure by asset class
        plt.subplot(3, 1, 3)
        _, net_exposure_by_class = self.get_exposure_by_asset_class()
        
        if not net_exposure_by_class.empty:
            # Sort by absolute value
            sorted_exposure = net_exposure_by_class.sort_values(key=abs, ascending=False)
            
            # Plot with color indicating direction
            colors = ['green' if x >= 0 else 'red' for x in sorted_exposure]
            sorted_exposure.plot(kind='barh', color=colors)
            
            plt.title('Net Exposure by Asset Class')
            plt.xlabel('Net Exposure')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            # Add text annotations with percentages
            for i, (cls, value) in enumerate(sorted_exposure.items()):
                plt.text(value + (0.01 if value >= 0 else -0.01), i, 
                         f"{value:.1%}", va='center', ha='left' if value >= 0 else 'right')
        else:
            plt.text(0.5, 0.5, 'No asset class exposure data available', 
                     ha='center', va='center', transform=plt.gca().transAxes)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
        # Create a second figure for position details
        plt.figure(figsize=(10, 8))
        
        # Group positions by asset class and direction
        positions_by_class = {}
        
        for asset_class, assets in self.asset_classes.items():
            valid_assets = [a for a in assets if a in self.weights.index]
            if valid_assets:
                long_assets = [a for a in valid_assets if self.weights[a] > 0]
                short_assets = [a for a in valid_assets if self.weights[a] < 0]
                
                if long_assets or short_assets:
                    positions_by_class[asset_class] = {
                        'long': {a: self.weights[a] for a in long_assets},
                        'short': {a: self.weights[a] for a in short_assets}
                    }
        
        # Plot positions by asset class
        for i, (asset_class, positions) in enumerate(positions_by_class.items()):
            long_positions = positions['long']
            short_positions = positions['short']
            
            # Only plot if we have positions
            if long_positions or short_positions:
                plt.subplot(len(positions_by_class), 1, i+1)
                
                # Combine positions into a single series
                all_positions = pd.Series({**long_positions, **short_positions})
                sorted_positions = all_positions.sort_values()
                
                # Plot with colors
                colors = ['red' if p < 0 else 'green' for p in sorted_positions]
                sorted_positions.plot(kind='barh', color=colors)
                
                plt.title(f'{asset_class} Positions')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
                
                # Add exposure information
                long_sum = sum(long_positions.values())
                short_sum = sum(short_positions.values())
                net_sum = long_sum + short_sum
                
                plt.figtext(0.01, 0.99 - 0.99 * i / len(positions_by_class), 
                         f"{asset_class}: Long {long_sum:.1%}, Short {abs(short_sum):.1%}, Net {net_sum:.1%}", 
                         va='top', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def save_allocation(self, filepath='data/balanced_allocation.csv'):
        """
        Save portfolio allocation to a CSV file with detailed information.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the allocation (default: 'data/balanced_allocation.csv')
        
        Returns:
        --------
        None
        """
        if self.weights is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create a DataFrame with weights and other information
            allocation_df = pd.DataFrame({
                'weight': self.weights,
                'direction': np.sign(self.weights),
                'strength': self.trend_strengths,
                'original_direction': self.trend_directions
            })
            
            # Add asset class information
            asset_class_map = {}
            for asset_class, assets in self.asset_classes.items():
                for asset in assets:
                    asset_class_map[asset] = asset_class
            
            allocation_df['asset_class'] = allocation_df.index.map(lambda x: asset_class_map.get(x, 'Unknown'))
            
            # Add reasons for direction changes if available
            if self.allocation_details and 'reasons' in self.allocation_details:
                reasons = self.allocation_details['reasons']
                allocation_df['reason'] = allocation_df.index.map(lambda x: reasons.get(x, ''))
            
            # Sort by asset class and weight
            allocation_df = allocation_df.sort_values(['asset_class', 'weight'], ascending=[True, False])
            
            # Save to CSV
            allocation_df.to_csv(filepath)
            
            # Save summary information
            summary = self.get_allocation_summary()
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(filepath.replace('.csv', '_summary.csv'), index=False)
            
            print(f"Allocation saved to {filepath}")
        else:
            print("No allocation to save. Run allocate_portfolio() first.")
    
    def load_allocation(self, filepath='data/balanced_allocation.csv'):
        """
        Load portfolio allocation from a CSV file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to load the allocation from (default: 'data/balanced_allocation.csv')
        
        Returns:
        --------
        pd.Series
            Portfolio weights
        """
        try:
            # Load allocation
            allocation_df = pd.read_csv(filepath, index_col=0)
            
            # Extract weights
            self.weights = allocation_df['weight']
            
            # Extract other information if available
            if 'original_direction' in allocation_df.columns:
                self.trend_directions = allocation_df['original_direction']
            
            if 'strength' in allocation_df.columns:
                self.trend_strengths = allocation_df['strength']
            
            # Extract asset classes
            if 'asset_class' in allocation_df.columns:
                self.asset_classes = {}
                for asset_class in allocation_df['asset_class'].unique():
                    assets = allocation_df[allocation_df['asset_class'] == asset_class].index.tolist()
                    self.asset_classes[asset_class] = assets
            
            # Try to load summary
            summary_path = filepath.replace('.csv', '_summary.csv')
            if os.path.exists(summary_path):
                summary_df = pd.read_csv(summary_path)
                
                # Extract regime if available
                if 'regime' in summary_df.columns:
                    self.regime = summary_df['regime'].iloc[0]
            
            print(f"Allocation loaded from {filepath}")
            return self.weights
        except FileNotFoundError:
            print(f"Allocation file {filepath} not found.")
            return None
        except Exception as e:
            print(f"Error loading allocation: {str(e)}")
            return None