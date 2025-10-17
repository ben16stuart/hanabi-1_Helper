import yfinance as yf
import pandas as pd
import os
import sys
from pathlib import Path

def download_stock_data(ticker):
    """Download hourly stock data for the specified ticker"""
    
    # Config
    period = "90d"
    interval = "1h"
    
    # Determine base path
    base_path = str(Path.home() / "Desktop" / "hanabi")
    output_file = f"{base_path}/{ticker}/hourly_data.csv"
    
    # Ensure stock directory exists
    stock_dir = f"{base_path}/{ticker}"
    os.makedirs(stock_dir, exist_ok=True)
    
    print(f"📥 Downloading {ticker} data...")
    print(f"   Period: {period}")
    print(f"   Interval: {interval}")
    
    try:
        # Download historical hourly data
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"❌ No data returned for {ticker}. Please check the ticker symbol.")
            return False
        
        # Drop missing rows
        df.dropna(inplace=True)
        
        if df.empty:
            print(f"❌ All data contained missing values for {ticker}")
            return False
        
        # Reset index to get datetime column
        df = df.reset_index()
        
        # Convert datetime to Unix timestamp in milliseconds
        df['Timestamp'] = df['Datetime'].astype('int64') // 10**6  # milliseconds
        
        # Final format: Timestamp, Price, Volume, High, Low, Open, Close
        df_formatted = pd.DataFrame({
            'Timestamp': df['Timestamp'],
            'Price': df['Close'].values.flatten(),   # Avoid 2D shape
            'Volume': df['Volume'].values.flatten(),
            'High': df['High'].values.flatten(),
            'Low': df['Low'].values.flatten(),
            'Open': df['Open'].values.flatten(),
            'Close': df['Close'].values.flatten()
        })
        
        # Save to CSV
        df_formatted.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(df_formatted)} rows to {output_file}")
        print(f"📊 Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        print(f"💰 Latest price: ${df_formatted['Price'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data for {ticker}: {str(e)}")
        return False


if __name__ == "__main__":
    # Check if ticker was provided as command line argument
    if len(sys.argv) < 2:
        print("❌ Error: No ticker symbol provided")
        print("Usage: python download_data.py TICKER")
        print("Example: python download_data.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    print(f"\n{'='*60}")
    print(f"Stock Data Downloader")
    print(f"{'='*60}\n")
    
    success = download_stock_data(ticker)
    
    print(f"\n{'='*60}\n")
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)