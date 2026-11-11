
from pathlib import Path
import fastf1
import pandas as pd

# Store the cache inside our project folder
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

fastf1.Cache.configure(cache_dir = str(CACHE_DIR))

# Connect to the historical F1 results API
ergast = fastf1.ergast.Ergast()

# Get results from the first race of 2024
results = ergast.get_race_results(
    season=2024,
    round=1,
    result_type="pandas"
)

# Display the results
df = results.content[0]
print(df.head())

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nTotal driver results:", len(df))
