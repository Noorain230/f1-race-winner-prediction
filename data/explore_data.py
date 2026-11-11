
from pathlib import Path
import pandas as pd

# Load the dataset
DATA_DIR = Path(__file__).resolve().parent
file_path = DATA_DIR / "f1_race_results.csv"

df = pd.read_csv(file_path)

print("===== DATASET OVERVIEW =====")
print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE RECORDS =====")
duplicates = df.duplicated(
    subset=["year", "round", "driverId"]
).sum()
print("Duplicate driver-race records:", duplicates)

print("\n===== RACE VALIDATION =====")
race_summary = df.groupby(["year", "round"]).agg(
    race_name=("race_name", "first"),
    drivers=("driverId", "nunique"),
    winners=("target_win", "sum")
).reset_index()

print("Total races:", len(race_summary))
print("\nNumber of drivers per race:")
print(race_summary["drivers"].value_counts().sort_index())

print("\nRaces without exactly one winner:")
print(race_summary[race_summary["winners"] != 1])

print("\n===== TARGET DISTRIBUTION =====")
print(df["target_win"].value_counts())
print("\nWinner percentage:")
print(round(df["target_win"].mean() * 100, 2), "%")

print("\n===== SAMPLE DATA =====")
print(df.head(10).to_string(index=False))
