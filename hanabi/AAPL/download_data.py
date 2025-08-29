import yfinance as yf
import pandas as pd
import os

# Config
ticker = "AAPL"
period = "90d"
interval = "1h"
output_dir = "hanabi-1"
output_file = f"<output location>/hanabi/{ticker}/hourly_data.csv"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Download historical hourly data
df = yf.download(ticker, period=period, interval=interval)

# Drop missing rows
df.dropna(inplace=True)

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
