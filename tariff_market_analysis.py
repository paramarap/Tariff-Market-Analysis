# Import necessary libraries for data manipulation, database handling, and fetching market data
import pandas as pd  # For handling data in table format (DataFrames)
import numpy as np  # For numerical operations (e.g., RSI calculation)
import sqlite3  # For storing data in a lightweight SQLite database
from datetime import datetime, timedelta  # For working with dates and time intervals
import pandas_datareader.data as web  # For fetching stock market data from online sources
import time  # For adding delays in retry logic during data fetching

# Define a list of tariff events with detailed descriptions
# Each event includes the year, a descriptive event name, the announcement date, and affected countries
events = [
    {
        "year": 2018,
        "event": "Steel and Aluminum Tariffs (25% on Steel, 10% on Aluminum)",
        "date": "2018-03-01",
        "country": "Multiple Countries (Global, with some exemptions later)"
    },  # U.S. imposed tariffs on steel and aluminum imports from various countries
    {
        "year": 2018,
        "event": "China Tariffs Phase 1 ($34 Billion on Goods)",
        "date": "2018-07-06",
        "country": "China"
    },  # First wave of tariffs targeting $34 billion of Chinese goods in the U.S.-China trade war
    {
        "year": 2018,
        "event": "China Tariffs Phase 2 ($200 Billion Additional Goods)",
        "date": "2018-09-17",
        "country": "China"
    },  # Expanded tariffs on $200 billion more of Chinese imports, escalating trade tensions
    {
        "year": 2019,
        "event": "China Tariff Increase (25% on $200 Billion Goods)",
        "date": "2019-05-10",
        "country": "China"
    },  # Increased tariff rate from 10% to 25% on previously targeted $200 billion of Chinese goods
    {
        "year": 2025,
        "event": "Trump 2025 Proposed Tariffs (Potential 10-20% on Imports)",
        "date": "2025-03-03",
        "country": "Canada, Mexico, and China"
    }  # Hypothetical future tariffs based on proposed policies, targeting major trading partners
]

# Add key date points for each event to analyze market reactions over time
for event in events:
    # Convert the announcement date string to a datetime object for calculations
    date = datetime.strptime(event["date"], "%Y-%m-%d")
    # Store the original announcement date
    event["announcement_date"] = event["date"]
    # Calculate and store dates for analysis periods relative to the announcement
    event["one_week_after"] = (date + timedelta(days=7)).strftime("%Y-%m-%d")  # 1 week post-announcement
    event["one_month_after"] = (date + timedelta(days=30)).strftime("%Y-%m-%d")  # 1 month post-announcement
    event["three_months_after"] = (date + timedelta(days=90)).strftime("%Y-%m-%d")  # 3 months post-announcement
    event["six_months_after"] = (date + timedelta(days=180)).strftime("%Y-%m-%d")  # 6 months post-announcement
    event["end_of_year"] = f"{event['year']}-12-31"  # End of the announcement year (YTD)

# Define a function to calculate the Relative Strength Index (RSI), a momentum indicator
def calculate_rsi(data, periods=14):
    """
    Calculate RSI to measure the speed and change of price movements.
    RSI ranges from 0 to 100; above 70 indicates overbought, below 30 indicates oversold.
    """
    delta = data.diff()  # Calculate daily price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()  # Average gains over 14 days
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()  # Average losses over 14 days
    rs = gain / loss  # Relative strength (gain/loss ratio)
    return 100 - (100 / (1 + rs))  # Convert to RSI formula

# Define a function to fetch market data from online sources with retry logic
def fetch_market_data(symbol, start, end, attempts=3):
    """
    Fetch historical stock data (e.g., SPY) from Stooq or Yahoo Finance with retries on failure.
    Parameters: symbol (e.g., 'SPY'), start date, end date, number of retry attempts.
    """
    for attempt in range(attempts):
        try:
            # Ensure dates are in string format for the data source
            if isinstance(start, datetime):
                start = start.strftime('%Y-%m-%d')
            if isinstance(end, datetime):
                end = end.strftime('%Y-%m-%d')
                
            print(f"Fetching {symbol} data from {start} to {end}...")
            # Try Stooq first (good for historical data)
            df = web.DataReader(symbol, 'stooq', start, end)
            
            if not df.empty:
                df = df.sort_index()  # Sort data by date
                print(f"Successfully fetched {len(df)} days of data for {symbol}")
                return df
            
            # Fallback to Yahoo Finance if Stooq fails
            df = web.DataReader(symbol, 'yahoo', start, end)
            if not df.empty:
                df = df.sort_index()
                print(f"Successfully fetched {len(df)} days of data for {symbol}")
                return df
                
            print(f"Warning: No data returned for {symbol} between {start} and {end}")
            return pd.DataFrame()  # Return empty DataFrame if no data found
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{attempts} - Error fetching data for {symbol}: {e}")
            if attempt < attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff (e.g., 1s, 2s, 4s)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch data for {symbol} after {attempts} attempts")
                return pd.DataFrame()

# Define a function to analyze market reactions to each tariff event
def analyze_event(event, symbol):
    """
    Analyze S&P 500 (SPY) reactions to tariff events, calculating price change, volume change, and RSI.
    Parameters: event (dict with event details), symbol (stock ticker, e.g., 'SPY').
    Returns: dict with analysis results for each time period.
    """
    try:
        # Convert announcement date to datetime for comparison
        start_date = datetime.strptime(event["date"], "%Y-%m-%d")
        lookback_start = start_date - timedelta(days=30)  # 30-day lookback for RSI calculation
        end_date = min(start_date + timedelta(days=365), datetime.now())  # Up to 1 year or current date
        
        # Skip events too recent to have meaningful data (within 30 days of today)
        if start_date > datetime.now() - timedelta(days=30):
            print(f"Skipping {event['event']} ({event['date']}) as it's too recent")
            return create_empty_result(event)
        
        # Fetch SPY data for the event period
        df = fetch_market_data(symbol, lookback_start, end_date)
        
        if df.empty:
            print(f"No data available for {symbol} for event {event['event']} ({event['date']})")
            return create_empty_result(event)
            
        # Standardize column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate RSI for the dataset
        df['rsi'] = calculate_rsi(df['close'])
        
        # Find the closest trading day to the announcement date
        closest_start = df.index[df.index >= start_date][0]
        start_data = df.loc[closest_start]
        pre_tariff_price = start_data['close']  # Baseline price on announcement day
        pre_tariff_volume = start_data['volume']  # Baseline volume on announcement day
        
        # Helper function to calculate metrics for a given date
        def calc_metrics(after_date):
            """Calculate price change %, volume change %, and RSI for a specific date."""
            if any(df.index >= after_date):
                closest_date = df.index[df.index >= after_date][0]
                after_data = df.loc[closest_date]
                
                # Calculate percentage changes relative to announcement day
                price_change = ((after_data['close'] - pre_tariff_price) / pre_tariff_price) * 100
                volume_change = ((after_data['volume'] - pre_tariff_volume) / pre_tariff_volume) * 100
                
                return {
                    'price_change_%': price_change,  # % change in price
                    'volume_change_%': volume_change,  # % change in trading volume
                    'rsi': after_data['rsi']  # RSI value at that date
                }
            return None

        # Calculate metrics for each time period
        metrics = {}
        for period in ['one_week_after', 'one_month_after', 'three_months_after', 
                      'six_months_after', 'end_of_year']:
            period_date = datetime.strptime(event[period], "%Y-%m-%d")
            period_metrics = calc_metrics(period_date)
            if period_metrics:
                for metric_name, value in period_metrics.items():
                    metrics[f"{period}_{metric_name}"] = value
            else:
                # If no data for the period (e.g., future dates), set to None
                for metric_name in ['price_change_%', 'volume_change_%', 'rsi']:
                    metrics[f"{period}_{metric_name}"] = None

        # Return the event details with calculated metrics
        return {
            "year": event["year"],
            "event": event["event"],
            "date": event["date"],
            **metrics
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol} for event {event['event']} ({event['date']}): {e}")
        return create_empty_result(event)

# Define a function to create an empty result for failed or future events
def create_empty_result(event):
    """
    Create a placeholder result with None values for events with no data.
    Parameters: event (dict with event details).
    Returns: dict with event info and null metrics.
    """
    periods = ['one_week_after', 'one_month_after', 'three_months_after', 
              'six_months_after', 'end_of_year']
    metrics = ['price_change_%', 'volume_change_%', 'rsi']
    
    result = {
        "year": event["year"],
        "event": event["event"],
        "date": event["date"]
    }
    
    # Set all metrics to None for each period
    for period in periods:
        for metric in metrics:
            result[f"{period}_{metric}"] = None
            
    return result

# Main execution: Analyze S&P 500 data for all tariff events
print("Fetching and analyzing simplified S&P 500 market data...")
sp500_data = [analyze_event(event, "SPY") for event in events]  # Process each event with SPY ticker

# Convert the list of results into a pandas DataFrame for easy handling
sp500_df = pd.DataFrame(sp500_data)

# Store the results in an SQLite database for future querying
conn = sqlite3.connect("tariff_analysis.db")
sp500_df.to_sql("sp500", conn, if_exists="replace", index=False)  # Overwrite existing table
conn.close()

# Save the results to a CSV file for public sharing and Excel analysis
sp500_df.to_csv("sp500_analysis.csv", index=False)

# Display a preview of key columns in the console
print("\nS&P 500 Analysis Table (Selected Columns):")
print(sp500_df[['year', 'event', 'date', 
                'one_week_after_price_change_%', 'one_month_after_price_change_%', 
                'three_months_after_price_change_%', 'six_months_after_price_change_%', 
                'end_of_year_price_change_%']])

# Confirm completion
print("\nAnalysis complete. Results saved to 'sp500_analysis.csv' and SQLite database 'tariff_analysis.db'.")
