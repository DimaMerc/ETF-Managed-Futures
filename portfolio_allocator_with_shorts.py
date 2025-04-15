"""
Enhanced portfolio allocation that ensures the strategy includes short positions.
"""
import pandas as pd


def allocate_portfolio(self):
    """
    Dynamic portfolio allocation based on signals and constraints with improved short positioning.
    """
    print("Allocating portfolio with enhanced short signal capability...")
    
    # Initialize with zeros
    weights = pd.Series(0.0, index=self.trend_directions.index)
    
    # Determine asset class allocations based on regime
    allocation_weights = self.determine_regime_allocations()
    
    # Display regime and allocations
    if self.regime:
        print(f"Current market regime: {self.regime}")
    print("Asset class allocations:")
    for asset_class, weight in allocation_weights.items():
        print(f"  {asset_class}: {weight:.2%}")
    
    # For each asset class, explicitly ensure some short positions
    for asset_class, class_weight in allocation_weights.items():
        assets = self.asset_classes[asset_class]
        
        if not assets:
            continue
        
        # Get signals for this class
        valid_assets = [a for a in assets if a in self.trend_strengths.index]
        if not valid_assets:
            continue
            
        # Calculate signal strength and direction
        strengths = self.trend_strengths[valid_assets]
        directions = self.trend_directions[valid_assets].copy()  # Make a copy to avoid modifying original
        
        # Force some negative signals for diversification
        if len(valid_assets) > 1:
            # Sort assets by strength (regardless of direction)
            sorted_strengths = strengths.sort_values()
            
            # Take bottom third of assets and force negative direction
            # This ensures we always have some short positions
            num_shorts = max(1, len(valid_assets) // 3)
            for asset in sorted_strengths.index[:num_shorts]:
                directions[asset] = -1  # Force negative direction
        
        # Normalize strengths within asset class
        norm_strengths = strengths / strengths.sum() if strengths.sum() > 0 else pd.Series(1.0/len(valid_assets), index=valid_assets)
        
        # Split allocation for long and short positions
        long_assets = [a for a in valid_assets if directions[a] > 0]
        short_assets = [a for a in valid_assets if directions[a] < 0]
        
        # Default allocation: 70% to longs, 30% to shorts
        long_allocation = 0.7
        short_allocation = 0.3
        
        # Adjust based on regime - increase shorts during uncertainty/tension
        if self.regime and ('Tension' in self.regime or 'Uncertainty' in self.regime):
            long_allocation = 0.6
            short_allocation = 0.4
        
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
        
        # If we have no short assets, force at least one
        if not short_assets and long_assets:
            # Take the weakest long asset and make it a short
            weakest_asset = strengths[long_assets].idxmin()
            # Remove from long allocation
            weights[weakest_asset] = 0
            # Add as short
            weights[weakest_asset] = -class_weight * short_allocation
    
    # Apply leverage
    weights = weights * self.leverage
    
    # Apply risk management constraints
    # No single asset can be more than max_position_size of portfolio
    weights = weights.clip(lower=-self.max_position_size, upper=self.max_position_size)
    
    # Ensure weights sum to leverage (if we have any positions)
    if abs(weights.sum()) > 0:
        weights = weights / abs(weights.sum()) * self.leverage
    
    # Store the weights
    self.weights = weights
    
    # Print long-short exposure
    long_exposure, short_exposure = self.get_long_short_exposure()
    print(f"Long exposure: {long_exposure:.2%}, Short exposure: {short_exposure:.2%}")
    
    print("Portfolio allocation complete")
    return self.weights
