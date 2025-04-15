"""
Advanced news sentiment analysis module focused on tariffs and trade tensions.
Uses NLP techniques to extract sentiment and potential market impact from news.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers for advanced NLP, gracefully handle if not available
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Using VADER sentiment analyzer only.")

class TariffSentimentAnalyzer:
    """
    Class for analyzing news sentiment related to tariffs and trade policy,
    with a focus on potential market impacts.
    """
    
    def __init__(self, use_advanced_nlp=True):
        """
        Initialize the TariffSentimentAnalyzer.
        
        Parameters:
        -----------
        use_advanced_nlp : bool, optional
            Whether to use advanced NLP models if available (default: True)
        """
        self.use_advanced_nlp = use_advanced_nlp and TRANSFORMERS_AVAILABLE
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.advanced_analyzer = None
        self.news_data = None
        self.sentiment_data = None
        self.impact_sectors = None
        
        # Set up advanced NLP if requested and available
        if self.use_advanced_nlp:
            try:
                # Initialize the financial sentiment analysis pipeline
                self.advanced_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                print("Advanced NLP model loaded successfully.")
            except Exception as e:
                print(f"Error loading advanced NLP model: {str(e)}")
                self.use_advanced_nlp = False
        
        # Define key tariff-related terms for keyword extraction
        self.tariff_keywords = [
            'tariff', 'tariffs', 'trade war', 'trade dispute', 'trade tension', 
            'trade policy', 'protectionism', 'protectionist', 'import duty', 
            'export control', 'economic sanction', 'customs duty', 'trade deal', 
            'trade agreement', 'retaliatory measures', 'trade deficit'
        ]
        
        # Define asset classes and sectors potentially affected by tariffs
        self.asset_impacts = {
            'Commodities': {
                'Steel': ['steel', 'iron', 'metal', 'aluminum'],
                'Agriculture': ['soybean', 'corn', 'wheat', 'pork', 'farm', 'agricultural'],
                'Energy': ['oil', 'gas', 'energy', 'fuel'],
                'Industrial Metals': ['copper', 'aluminum', 'nickel', 'zinc']
            },
            'Currencies': {
                'USD': ['dollar', 'usd', 'greenback', 'us currency'],
                'CNY': ['yuan', 'renminbi', 'chinese currency', 'cny'],
                'EUR': ['euro', 'eur', 'european currency'],
                'Emerging': ['emerging market', 'developing economy']
            },
            'Equities': {
                'Technology': ['tech', 'semiconductor', 'computer', 'electronics'],
                'Automotive': ['auto', 'car', 'vehicle', 'automotive'],
                'Retail': ['retail', 'consumer goods', 'ecommerce'],
                'Manufacturing': ['manufacturing', 'factory', 'industrial']
            }
        }
    
    def analyze_sentiment(self):
        """
        Analyze sentiment of collected news.
        
        Returns:
        --------
        pd.DataFrame
            Sentiment analysis results
        """
        if self.news_data is None or len(self.news_data) == 0:
            print("No news data available. Call collect_news() first or provide news_data directly.")
            return None
        
        print("Analyzing sentiment of news articles...")
        
        # Initialize results DataFrame
        results = self.news_data.copy()
        
        # Add columns for sentiment scores
        results['vader_compound'] = 0.0
        results['vader_sentiment'] = ''
        
        if self.use_advanced_nlp:
            results['advanced_score'] = 0.0
            results['advanced_sentiment'] = ''
        
        # Process each article
        for i, row in results.iterrows():
            # Combine headline and content for analysis
            text = f"{row['headline']}. {row['content']}"
            
            # 1. VADER Sentiment Analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results.at[i, 'vader_compound'] = vader_scores['compound']
            
            # Map scores to sentiment labels
            if vader_scores['compound'] >= 0.05:
                results.at[i, 'vader_sentiment'] = 'positive'
            elif vader_scores['compound'] <= -0.05:
                results.at[i, 'vader_sentiment'] = 'negative'
            else:
                results.at[i, 'vader_sentiment'] = 'neutral'
            
            # 2. Advanced NLP (if available)
            if self.use_advanced_nlp and self.advanced_analyzer is not None:
                try:
                    # Truncate text if needed (transformer models often have max length)
                    if len(text) > 512:
                        text = text[:512]
                    
                    # Get sentiment prediction
                    advanced_result = self.advanced_analyzer(text)[0]
                    label = advanced_result['label']
                    score = advanced_result['score']
                    
                    results.at[i, 'advanced_sentiment'] = label.lower()
                    results.at[i, 'advanced_score'] = score if label == 'positive' else -score if label == 'negative' else 0
                except Exception as e:
                    print(f"Error in advanced sentiment analysis: {str(e)}")
                    results.at[i, 'advanced_sentiment'] = results.at[i, 'vader_sentiment']
                    results.at[i, 'advanced_score'] = results.at[i, 'vader_compound']
        
        # Combine sentiment scores
        if self.use_advanced_nlp:
            # Average the normalized scores
            results['combined_score'] = (results['vader_compound'] + results['advanced_score']) / 2
        else:
            results['combined_score'] = results['vader_compound']
        
        # Final sentiment label based on combined score
        results['sentiment'] = results['combined_score'].apply(
            lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')
        
        self.sentiment_data = results
        
        # Analyze sentiment statistics
        positive_count = (results['sentiment'] == 'positive').sum()
        negative_count = (results['sentiment'] == 'negative').sum()
        neutral_count = (results['sentiment'] == 'neutral').sum()
        
        print(f"Sentiment analysis complete: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
        return results
    
    def extract_impact_sectors(self):
        """
        Extract sectors and assets potentially impacted by tariffs mentioned in news.
        
        Returns:
        --------
        pd.DataFrame
            Impact analysis by sector
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return None
        
        print("Analyzing potential market impacts by sector...")
        
        # Initialize impact tracking
        impact_data = {
            'date': [],
            'asset_class': [],
            'sector': [],
            'sentiment': [],
            'score': [],
            'headline': []
        }
        
        # Process each article
        for i, row in self.sentiment_data.iterrows():
            text = f"{row['headline'].lower()} {row['content'].lower()}"
            date = row['date']
            sentiment = row['sentiment']
            score = row['combined_score']
            headline = row['headline']
            
            # Check for mentions of each sector
            mentions_found = False
            
            for asset_class, sectors in self.asset_impacts.items():
                for sector, keywords in sectors.items():
                    # Check if any keyword is mentioned
                    for keyword in keywords:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                            impact_data['date'].append(date)
                            impact_data['asset_class'].append(asset_class)
                            impact_data['sector'].append(sector)
                            impact_data['sentiment'].append(sentiment)
                            impact_data['score'].append(score)
                            impact_data['headline'].append(headline)
                            mentions_found = True
                            break
                    
                    if mentions_found:
                        break
            
            # If no specific sector was mentioned, add to general impact
            if not mentions_found:
                impact_data['date'].append(date)
                impact_data['asset_class'].append('General')
                impact_data['sector'].append('General')
                impact_data['sentiment'].append(sentiment)
                impact_data['score'].append(score)
                impact_data['headline'].append(headline)
        
        # Create DataFrame
        impact_df = pd.DataFrame(impact_data)
        
        # Store for later use
        self.impact_sectors = impact_df
        
        # Summary statistics
        impact_counts = impact_df.groupby(['asset_class', 'sector']).size()
        impact_sentiment = impact_df.groupby(['asset_class', 'sector'])['score'].mean()
        
        print("Impact analysis by sector complete")
        return impact_df
    
    def calculate_daily_sentiment(self):
        """
        Calculate daily aggregate sentiment scores.
        
        Returns:
        --------
        pd.DataFrame
            Daily sentiment scores
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return None
        
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(self.sentiment_data['date']):
            self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
        
        # Group by date and calculate statistics
        daily_sentiment = self.sentiment_data.groupby(self.sentiment_data['date'].dt.date).agg({
            'combined_score': ['mean', 'count', 'std'],
            'sentiment': lambda x: x.value_counts().to_dict()
        })
        
        # Flatten the columns
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        
        # Add positive and negative counts
        for sent in ['positive', 'negative', 'neutral']:
            daily_sentiment[f'{sent}_count'] = daily_sentiment['sentiment_<lambda>'].apply(
                lambda x: x.get(sent, 0) if isinstance(x, dict) else 0)
        
        # Calculate positive-negative ratio
        daily_sentiment['sentiment_ratio'] = daily_sentiment.apply(
            lambda x: x['positive_count'] / max(1, x['negative_count']), axis=1)
        
        # Convert index to datetime
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Sort by date
        daily_sentiment = daily_sentiment.sort_index()
        
        # Forward fill to get values for dates without news
        full_date_range = pd.date_range(daily_sentiment.index.min(), daily_sentiment.index.max())
        daily_sentiment = daily_sentiment.reindex(full_date_range).fillna(method='ffill')
        
        return daily_sentiment
    
    def calculate_sector_sentiment(self):
        """
        Calculate sentiment scores aggregated by asset class and sector.
        
        Returns:
        --------
        pd.DataFrame
            Sentiment scores by sector
        """
        if self.impact_sectors is None:
            print("No impact sector data available. Call extract_impact_sectors() first.")
            return None
        
        # Aggregate sentiment by asset class and sector
        sector_sentiment = self.impact_sectors.groupby(['asset_class', 'sector']).agg({
            'score': ['mean', 'count', 'std'],
            'sentiment': lambda x: pd.Series(x).value_counts().to_dict()
        })
        
        # Flatten the columns
        sector_sentiment.columns = ['_'.join(col).strip() for col in sector_sentiment.columns.values]
        
        # Add positive and negative counts
        for sent in ['positive', 'negative', 'neutral']:
            sector_sentiment[f'{sent}_count'] = sector_sentiment['sentiment_<lambda>'].apply(
                lambda x: x.get(sent, 0) if isinstance(x, dict) else 0)
        
        # Calculate positive-negative ratio
        sector_sentiment['sentiment_ratio'] = sector_sentiment.apply(
            lambda x: x['positive_count'] / max(1, x['negative_count']), axis=1)
        
        return sector_sentiment
    
    def get_tariff_indicators(self, include_sector_adjustments=True):
        """
        Generate indicators for strategy adjustment based on tariff news sentiment.
        
        Parameters:
        -----------
        include_sector_adjustments : bool, optional
            Whether to include sector-specific adjustments (default: True)
        
        Returns:
        --------
        dict
            Tariff sentiment indicators for strategy adjustment
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return None
        
        # Calculate daily sentiment
        daily_sentiment = self.calculate_daily_sentiment()
        
        # Get the latest 7 days of sentiment data
        recent_sentiment = daily_sentiment.iloc[-7:]
        
        # Calculate overall tariff tension indicator (0-100)
        recent_scores = recent_sentiment['combined_score_mean']
        
        # Normalize the sentiment scores to a 0-100 scale
        # Empirically, most sentiment scores are in the -0.5 to 0.5 range
        # We'll map this to 0-100 with negative sentiment (trade tension) = high values
        normalized_scores = 50 - (recent_scores * 100)  # Negative sentiment = high tension
        normalized_scores = normalized_scores.clip(0, 100)
        
        # Calculate the weighted average, giving more weight to recent days
        weights = np.exp(np.linspace(0, 1, len(normalized_scores)))
        weights = weights / weights.sum()
        tariff_tension = float((normalized_scores * weights).sum())
        
        # Calculate trend - is tension increasing or decreasing?
        if len(normalized_scores) > 1:
            first_half = normalized_scores.iloc[:len(normalized_scores)//2].mean()
            second_half = normalized_scores.iloc[len(normalized_scores)//2:].mean()
            tension_trend = second_half - first_half
        else:
            tension_trend = 0
        
        # Initialize results dictionary
        indicators = {
            'tariff_tension': tariff_tension,
            'tension_trend': tension_trend,
            'recent_sentiment': recent_scores.mean(),
            'sentiment_count': recent_sentiment['combined_score_count'].sum(),
            'sectors': {}
        }
        
        # Add sector-specific adjustments if requested
        if include_sector_adjustments and self.impact_sectors is not None:
            sector_sentiment = self.calculate_sector_sentiment()
            
            # Convert sector sentiment to indicators
            for (asset_class, sector), row in sector_sentiment.iterrows():
                sentiment_value = row['score_mean']
                sentiment_count = row['score_count']
                
                # Only include sectors with sufficient mentions
                if sentiment_count >= 2:
                    # Normalize to a -5 to +5 scale for sector adjustments
                    # This is used to adjust the weight/direction of assets in each sector
                    adjustment = sentiment_value * 10
                    adjustment = max(-5, min(5, adjustment))
                    
                    if asset_class not in indicators['sectors']:
                        indicators['sectors'][asset_class] = {}
                    
                    indicators['sectors'][asset_class][sector] = {
                        'adjustment': adjustment,
                        'count': sentiment_count,
                        'sentiment': sentiment_value
                    }
        
        return indicators
    
    def adjust_strategy_weights(self, weights, trend_directions=None, trend_strengths=None):
        """
        Adjust strategy weights based on tariff sentiment indicators.
        
        Parameters:
        -----------
        weights : pd.Series
            Current portfolio weights
        trend_directions : pd.Series, optional
            Trend directions for each asset (default: None)
        trend_strengths : pd.Series, optional
            Trend strengths for each asset (default: None)
        
        Returns:
        --------
        pd.Series
            Adjusted portfolio weights
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return weights
        
        # Get tariff indicators
        indicators = self.get_tariff_indicators()
        
        if indicators is None:
            print("No valid tariff indicators. Using original weights.")
            return weights
        
        print(f"Adjusting strategy based on tariff sentiment (tension level: {indicators['tariff_tension']:.1f}/100)")
        
        # Create a copy of the weights
        adjusted_weights = weights.copy()
        
        # Define asset class and sector mappings
        asset_class_patterns = {
            'Commodities': [
                'WTI', 'BRENT', 'NATURAL_GAS', 'COPPER', 'ALUMINUM', 
                'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE'
            ],
            'Currencies': [
                'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF'
            ],
            'Bonds': [
                'TREASURY', 'BOND', 'YIELD'
            ],
            'Equities': [
                'SPY', 'QQQ', 'DIA', 'IWM', 'S&P', 'NASDAQ', 'DOW', 'RUSSELL'
            ]
        }
        
        sector_patterns = {
            'Steel': ['STEEL', 'IRON', 'METAL'],
            'Agriculture': ['WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE'],
            'Energy': ['WTI', 'BRENT', 'NATURAL_GAS', 'OIL'],
            'Industrial Metals': ['COPPER', 'ALUMINUM', 'NICKEL', 'ZINC'],
            'Technology': ['QQQ', 'NASDAQ'],
            'Automotive': [],  # No direct matches in typical futures
            'Manufacturing': ['INDUSTRIAL'],
            'USD': ['DOLLAR', 'USD'],
            'EUR': ['EUR', 'EURO'],
            'CNY': ['YUAN', 'CNY', 'RENMINBI']
        }
        
        # Apply overall tariff tension adjustment
        tariff_tension = indicators['tariff_tension']
        tension_factor = (tariff_tension - 50) / 50  # -1 to +1 scale, positive = high tension
        
        # General adjustment rules based on tariff tension:
        # 1. High tension -> Increase shorts in equities, increase longs in bonds
        # 2. High tension -> Adjust commodity exposure based on sector impacts
        # 3. High tension -> Adjust currency exposure (typically USD strengthens)
        
        for asset in weights.index:
            asset_upper = asset.upper()
            
            # Identify asset class
            asset_class = None
            for class_name, patterns in asset_class_patterns.items():
                if any(pattern in asset_upper for pattern in patterns):
                    asset_class = class_name
                    break
            
            if asset_class is None:
                continue  # Skip if we can't identify the asset class
            
            # Identify sector if applicable
            asset_sector = None
            for sector_name, patterns in sector_patterns.items():
                if any(pattern in asset_upper for pattern in patterns):
                    asset_sector = sector_name
                    break
            
            # Apply adjustments based on asset class and tariff tension
            
            # 1. Basic tension-based adjustments
            if asset_class == 'Equities':
                # High tension = more negative for equities
                if weights[asset] > 0:  # Long positions
                    adjusted_weights[asset] = weights[asset] * (1 - 0.3 * max(0, tension_factor))
                else:  # Short positions
                    adjusted_weights[asset] = weights[asset] * (1 + 0.3 * max(0, tension_factor))
            
            elif asset_class == 'Bonds':
                # High tension = more positive for bonds (flight to safety)
                if weights[asset] > 0:  # Long positions
                    adjusted_weights[asset] = weights[asset] * (1 + 0.3 * max(0, tension_factor))
                else:  # Short positions
                    adjusted_weights[asset] = weights[asset] * (1 - 0.3 * max(0, tension_factor))
            
            elif asset_class == 'Currencies':
                # USD typically strengthens in high tension
                if 'USD' in asset_upper or 'DOLLAR' in asset_upper:
                    if weights[asset] > 0:  # Long positions
                        adjusted_weights[asset] = weights[asset] * (1 + 0.2 * max(0, tension_factor))
                    else:  # Short positions
                        adjusted_weights[asset] = weights[asset] * (1 - 0.2 * max(0, tension_factor))
            
            # 2. Sector-specific adjustments from news analysis
            if 'sectors' in indicators and asset_class in indicators['sectors']:
                sectors = indicators['sectors'][asset_class]
                
                # If we have sector information for this asset
                if asset_sector and asset_sector in sectors:
                    sector_adj = sectors[asset_sector]['adjustment'] / 10  # Convert -5:5 scale to -0.5:0.5
                    
                    # Apply sector adjustment based on current position direction
                    if weights[asset] > 0:  # Long positions
                        adjusted_weights[asset] = weights[asset] * (1 + sector_adj)
                    else:  # Short positions
                        adjusted_weights[asset] = weights[asset] * (1 - sector_adj)
            
            # 3. Adjust trend direction if provided
            if trend_directions is not None and trend_strengths is not None:
                if asset in trend_directions.index and asset in trend_strengths.index:
                    # High trade tension can potentially reverse or strengthen trends
                    if tension_factor > 0.5:  # Very high tension
                        # For equities, consider reversing positive trends
                        if asset_class == 'Equities' and trend_directions[asset] > 0:
                            trend_directions[asset] = -1  # Force negative
                            print(f"Reversing {asset} trend due to high tariff tension")
                        
                        # For bonds, consider reversing negative trends
                        if asset_class == 'Bonds' and trend_directions[asset] < 0:
                            trend_directions[asset] = 1  # Force positive
                            print(f"Reversing {asset} trend due to high tariff tension")
                        
                        # For specific commodities affected by tariffs, strengthen trends
                        if asset_class == 'Commodities' and asset_sector in ['Steel', 'Agriculture']:
                            trend_strengths[asset] *= 1.3  # Strengthen trend
                            print(f"Strengthening {asset} trend due to tariff impact on {asset_sector}")
        
        # Normalize the adjusted weights to maintain the same leverage
        if abs(weights.sum()) > 0:
            leverage = abs(weights.sum())
            adjusted_weights = adjusted_weights / abs(adjusted_weights.sum()) * leverage
        
        return adjusted_weights, trend_directions, trend_strengths
    
    def visualize_sentiment_analysis(self):
        """
        Visualize the results of sentiment analysis on tariff news.
        
        Returns:
        --------
        None
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return
        
        # Set up the figure
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Sentiment distribution
        plt.subplot(3, 1, 1)
        sentiment_counts = self.sentiment_data['sentiment'].value_counts()
        
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                     color=[colors.get(x, 'blue') for x in sentiment_counts.index])
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.title('Sentiment Distribution in Tariff-Related News')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment over time
        plt.subplot(3, 1, 2)
        
        try:
            # Calculate daily sentiment
            daily_sentiment = self.calculate_daily_sentiment()
            
            # Plot sentiment scores
            daily_sentiment['combined_score_mean'].plot(label='Average Sentiment', 
                                                       color='blue', linewidth=2)
            
            # Add a line at zero
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Color the background based on sentiment
            for i in range(len(daily_sentiment) - 1):
                score = daily_sentiment['combined_score_mean'].iloc[i]
                start = daily_sentiment.index[i]
                end = daily_sentiment.index[i+1]
                
                if score > 0.05:  # Positive
                    plt.axvspan(start, end, alpha=0.2, color='green')
                elif score < -0.05:  # Negative
                    plt.axvspan(start, end, alpha=0.2, color='red')
            
            plt.title('Sentiment Trend Over Time')
            plt.ylabel('Sentiment Score')
            plt.ylim(-1, 1)
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting sentiment over time: {str(e)}")
            plt.text(0.5, 0.5, 'Error plotting sentiment trend', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 3: Sector Impact
        plt.subplot(3, 1, 3)
        
        try:
            if self.impact_sectors is not None:
                # Calculate sector sentiment
                sector_sentiment = self.calculate_sector_sentiment()
                
                # Select a few key sectors to plot
                key_sectors = sector_sentiment.loc[sector_sentiment['score_count'] >= 2]
                
                if not key_sectors.empty:
                    # Plot as horizontal bar chart
                    # Create a new DataFrame for plotting
                    plot_data = pd.DataFrame({
                        'Sentiment': key_sectors['score_mean'],
                        'Count': key_sectors['score_count']
                    })
                    
                    # Create sector labels
                    plot_data['Sector'] = [f"{ac}: {sc}" for ac, sc in plot_data.index]
                    
                    # Sort by sentiment
                    plot_data = plot_data.sort_values('Sentiment')
                    
                    # Create horizontal bar chart
                    bars = plt.barh(plot_data['Sector'], plot_data['Sentiment'],
                                  color=[colors.get('positive' if x > 0 else 'negative' if x < 0 else 'neutral', 'blue') 
                                         for x in plot_data['Sentiment']])
                    
                    # Add count as text
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        direction = 1 if width >= 0 else -1
                        plt.text(width + 0.02 * direction, bar.get_y() + bar.get_height()/2.,
                                f'{plot_data["Count"].iloc[i]:.0f} mentions', va='center')
                    
                    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
                    plt.title('Sentiment by Sector and Asset Class')
                    plt.xlabel('Sentiment Score')
                    plt.xlim(-1, 1)
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'No sector data with sufficient mentions', 
                            ha='center', va='center', transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, 'No sector impact data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            print(f"Error plotting sector impact: {str(e)}")
            plt.text(0.5, 0.5, 'Error plotting sector impact', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Create a second figure for tariff tension indicator
        plt.figure(figsize=(10, 6))
        
        try:
            # Get tariff indicators
            indicators = self.get_tariff_indicators()
            
            if indicators is not None:
                # Set up the tension gauge
                tension = indicators['tariff_tension']
                trend = indicators['tension_trend']
                
                # Create a gauge-like visualization
                gauge_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
                
                # Main gauge 
                plt.bar([0], [tension], color=gauge_colors[int(tension)], width=0.8)
                
                # Set up gauge ticks
                plt.xticks([])
                plt.yticks([0, 25, 50, 75, 100], ['0\nLow Tension', '25', '50\nNeutral', '75', '100\nHigh Tension'])
                
                # Add trend arrow
                sign = '↑' if trend > 0 else '↓' if trend < 0 else '→'
                plt.text(0, tension + 5, f"{sign} {abs(trend):.1f}", ha='center', 
                        fontsize=14, weight='bold', 
                        color='red' if trend > 0 else 'green' if trend < 0 else 'gray')
                
                plt.title('Tariff Tension Indicator')
                plt.ylim(0, 110)
                plt.grid(True, alpha=0.3)
                
                # Add information text
                plt.figtext(0.5, 0.02, 
                          (f"Based on {indicators['sentiment_count']:.0f} news articles. "
                           f"Recent sentiment: {indicators['recent_sentiment']:.2f}"),
                          ha='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))
                
                # Add recommendations based on tension level
                if tension < 30:
                    trend_text = "positive"
                    recommend = "Low trade tension may favor risk assets (equities)."
                elif tension < 70:
                    trend_text = "neutral"
                    recommend = "Moderate trade tension - balanced allocation recommended."
                else:
                    trend_text = "negative"
                    recommend = "High trade tension - consider defensive positioning (bonds, gold)."
                
                plt.figtext(0.5, 0.91, recommend, ha='center', fontsize=12, 
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))
            else:
                plt.text(0.5, 0.5, 'No tariff indicators available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            print(f"Error creating tariff tension gauge: {str(e)}")
            plt.text(0.5, 0.5, 'Error creating tariff tension gauge', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def save_sentiment_data(self, filepath='data/tariff_sentiment.csv'):
        """
        Save sentiment analysis results to a CSV file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the sentiment data (default: 'data/tariff_sentiment.csv')
        
        Returns:
        --------
        None
        """
        if self.sentiment_data is None:
            print("No sentiment data available. Call analyze_sentiment() first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save sentiment data
        self.sentiment_data.to_csv(filepath, index=False)
        print(f"Sentiment data saved to {filepath}")
        
        # Save impact data if available
        if self.impact_sectors is not None:
            impact_path = filepath.replace('.csv', '_impacts.csv')
            self.impact_sectors.to_csv(impact_path, index=False)
            print(f"Impact sector data saved to {impact_path}")
        
        # Save daily sentiment
        daily_sentiment = self.calculate_daily_sentiment()
        if daily_sentiment is not None:
            daily_path = filepath.replace('.csv', '_daily.csv')
            daily_sentiment.to_csv(daily_path)
            print(f"Daily sentiment data saved to {daily_path}")
        
        # Save indicators
        indicators = self.get_tariff_indicators()
        if indicators is not None:
            # Convert to DataFrame for saving
            indicator_df = pd.DataFrame({
                'indicator': list(indicators.keys()),
                'value': [indicators[k] if not isinstance(indicators[k], dict) else str(indicators[k]) 
                          for k in indicators.keys()]
            })
            
            indicator_path = filepath.replace('.csv', '_indicators.csv')
            indicator_df.to_csv(indicator_path, index=False)
            print(f"Tariff indicators saved to {indicator_path}")
    
    def load_sentiment_data(self, filepath='data/tariff_sentiment.csv'):
        """
        Load sentiment analysis results from a CSV file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to load the sentiment data from (default: 'data/tariff_sentiment.csv')
        
        Returns:
        --------
        pd.DataFrame
            Loaded sentiment data
        """
        try:
            # Load sentiment data
            self.sentiment_data = pd.read_csv(filepath)
            print(f"Sentiment data loaded from {filepath}")
            
            # Convert date column to datetime
            if 'date' in self.sentiment_data.columns:
                self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
            
            # Load impact data if available
            impact_path = filepath.replace('.csv', '_impacts.csv')
            if os.path.exists(impact_path):
                self.impact_sectors = pd.read_csv(impact_path)
                print(f"Impact sector data loaded from {impact_path}")
                
                # Convert date column to datetime
                if 'date' in self.impact_sectors.columns:
                    self.impact_sectors['date'] = pd.to_datetime(self.impact_sectors['date'])
            
            return self.sentiment_data
        except FileNotFoundError:
            print(f"Sentiment file {filepath} not found.")
            return None
        except Exception as e:
            print(f"Error loading sentiment data: {str(e)}")
            return None
