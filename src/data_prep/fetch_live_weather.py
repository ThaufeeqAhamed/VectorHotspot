import pandas as pd
import requests
import json
import time
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_weather_batch(lats, lons):
    """Fetches weather from Open-Meteo for a batch of coordinates"""
    # Open-Meteo API URL requires comma-separated strings for multiple coordinates
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "Asia/Kolkata",
        "past_days": 42, # Fetch past 6 weeks to build lag features (t-1 to t-4)
        "forecast_days": 1
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching weather: {response.status_code} - {response.text}")
        return None

def process_live_weather(h3_grid_path, sample_size=1000):
    print(f"Loading H3 grid from {h3_grid_path}...")
    df = pd.read_csv(h3_grid_path)
    
    # We take a sample to avoid hitting rate limits instantly during prototype testing.
    # In a full enterprise deployment, you would run this across all 620,000 cells using a paid API tier.
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    lats = df_sample['center_lat'].tolist()
    lons = df_sample['center_lon'].tolist()
    
    # Open-meteo max batch size is usually around 100
    batch_size = 50
    all_weather_data = []
    
    print(f"Fetching live 2026 weather for {sample_size} cells...")
    
    for i in range(0, len(lats), batch_size):
        batch_lats = lats[i:i+batch_size]
        batch_lons = lons[i:i+batch_size]
        batch_h3 = df_sample['h3_index'].iloc[i:i+batch_size].tolist()
        
        data = fetch_weather_batch(batch_lats, batch_lons)
        if data:
            # If multiple coords are requested, data is a list of dicts. If 1, it's a dict.
            if isinstance(data, list):
                for j, loc_data in enumerate(data):
                    loc_data['h3_index'] = batch_h3[j]
                    all_weather_data.append(loc_data)
            else:
                data['h3_index'] = batch_h3[0]
                all_weather_data.append(data)
                
        time.sleep(1) # Rate limit protection for free API tier
        print(f"  Processed {min(i+batch_size, len(lats))}/{len(lats)}")
        
    # Save the raw JSON data
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    output_path = LIVE_DATA_DIR / f"live_weather_{today_str}.json"
    with open(output_path, 'w') as f:
        json.dump(all_weather_data, f)
        
    print(f"\nSuccessfully fetched and saved weather data to:")
    print(f"  {output_path}")
    print("This data can now be fed into our Environment-Only Model for LIVE 2026 inference!")

if __name__ == "__main__":
    h3_path = PROCESSED_DATA_DIR / "india_h3_grid_res7.csv"
    # Testing with 100 cells. For the whole country, change to 620000 
    # (Requires Open-Meteo Commercial License for >10k calls/day)
    process_live_weather(h3_path, sample_size=100)
