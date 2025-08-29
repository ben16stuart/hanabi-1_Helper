import requests
import pandas as pd

# Get the response
url = "https://api.alternative.me/fng/?limit=0&format=json"
response = requests.get(url)
response.raise_for_status()  # Raise error for bad responses

# Parse the JSON
data = response.json()

# Extract the data list
records = data['data']

# Convert to DataFrame
df = pd.DataFrame(records)

# Convert timestamp to readable date
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
df['timestamp'] = df['timestamp'].dt.strftime('%d-%m-%Y')
df = df.rename(columns={
    'value': 'fng_value',
    'value_classification': 'fng_classification',
    'timestamp': 'date'
})

# Reorder columns
df = df[['date', 'fng_value', 'fng_classification']]

# Print or save
df.to_csv("<path to>/hanabi/sentiment-fear-and-greed/fear_greed_data/fear_greed_index.csv", index=False, header=False)
print("✅ Saved Fear Greed File")
