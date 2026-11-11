
from pathlib import Path

import fastf1
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.configure(cache_dir = str(CACHE_DIR))

ergast = fastf1.ergast.Ergast()
SEASONS = [2023, 2024, 2025]
all_races = []

for year in SEASONS:
    print(f"\nCollecting {year} season...")

    schedule = ergast.get_race_schedule(season=year)

    for _, race_info in schedule.iterrows():
        round_number = int(race_info["round"])
        race_name = race_info["raceName"]

        try:
            response = ergast.get_race_results(
                season=year,
                round=round_number,
                result_type="pandas"
            )

            if not response.content or response.content[0].empty:
                print(f"  Skipping {race_name}: no results returned")
                continue

            df = response.content[0].copy()
            df["year"] = year
            df["round"] = round_number
            df["race_name"] = race_name

            df["position"] = pd.to_numeric(
                df["position"], errors="coerce"
            )
            df["grid"] = pd.to_numeric(
                df["grid"], errors="coerce"
            )

            df["target_win"] = (df["position"] == 1).astype(int)

            columns = [
                "year", "round", "race_name",
                "driverId", "givenName", "familyName",
                "constructorName", "grid", "position",
                "points", "target_win"
            ]

            df = df[[col for col in columns if col in df.columns]]
            all_races.append(df)

            print(f"  Round {round_number}: {race_name} — {len(df)} drivers")

        except Exception as error:
            print(f"  Error in round {round_number} ({race_name}): {error}")

if all_races:
    final_df = pd.concat(all_races, ignore_index=True)

    output_file = DATA_DIR / "f1_race_results.csv"
    final_df.to_csv(output_file, index=False)

    print("\nCollection finished.")
    print("Total driver-race records:", len(final_df))
    print("Unique races:", final_df[["year", "round"]].drop_duplicates().shape[0])
    print("\nRecords per season:")
    print(final_df.groupby("year")["round"].nunique())

    print("\nDriver counts per race:")
    print(final_df.groupby(["year", "round"])["driverId"].nunique())

    print("\nSaved to:", output_file)
else:
    print("No race results were collected.")
