
from pathlib import Path
import pandas as pd

# File paths
DATA_DIR = Path(__file__).resolve().parent
input_file = DATA_DIR / "f1_race_results.csv"
output_file = DATA_DIR / "f1_model_data.csv"

# Load and sort races chronologically
df = pd.read_csv(input_file)
df = df.sort_values(["year", "round"]).reset_index(drop=True)

# Make sure grid and result columns are numeric
df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df["points"] = pd.to_numeric(df["points"], errors="coerce")

# Count how many races each driver has entered previously
df["driver_prev_races"] = (
    df.groupby("driverId").cumcount()
)

# Average finishing position in previous races only
df["driver_prev_avg_position"] = (
    df.groupby("driverId")["position"]
      .transform(lambda s: s.shift(1).expanding().mean())
)

# Average points in previous races only
df["driver_prev_avg_points"] = (
    df.groupby("driverId")["points"]
      .transform(lambda s: s.shift(1).expanding().mean())
)

# Average finishing position over the previous five races
df["driver_prev_5_avg_position"] = (
    df.groupby("driverId")["position"]
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
)

# Number of wins before the current race
df["driver_prev_wins"] = (
    df.groupby("driverId")["target_win"]
      .transform(lambda s: s.shift(1).fillna(0).cumsum())
)

# Early races have no historical data, so use neutral/default values
df["driver_prev_avg_position"] = (
    df["driver_prev_avg_position"].fillna(10.5)
)
df["driver_prev_avg_points"] = (
    df["driver_prev_avg_points"].fillna(0)
)
df["driver_prev_5_avg_position"] = (
    df["driver_prev_5_avg_position"].fillna(10.5)
)

# Keep only features that could be known before the race result
feature_columns = [
    "year",
    "round",
    "race_name",
    "driverId",
    "givenName",
    "familyName",
    "constructorName",
    "grid",
    "driver_prev_races",
    "driver_prev_avg_position",
    "driver_prev_avg_points",
    "driver_prev_5_avg_position",
    "driver_prev_wins",
    "target_win",
]

model_df = df[feature_columns].copy()

model_df.to_csv(output_file, index=False)

print("Feature preparation complete!")
print("Dataset shape:", model_df.shape)
print("\nColumns:")
print(model_df.columns.tolist())

print("\nSample rows:")
print(model_df.head(10).to_string(index=False))

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nSaved to:", output_file)
