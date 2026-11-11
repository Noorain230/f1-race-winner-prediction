
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "f1_model_data.csv"
MODEL_PATH = BASE_DIR / "models" / "f1_logistic_model.joblib"

# 2. Page configuration
st.set_page_config(
    page_title="F1 Race Winner Predictor",
    page_icon="🏎️",
    layout="wide"
)

# 3. Title and description
st.title("F1 Race Winner Predictor")

st.write(
    "Explore historical Formula 1 races and see which driver "
    "your Machine Learning model ranks highest."
)

st.info(
    "This version analyses historical 2025 races that were held "
    "out of the model's training data. Scores are model estimates, "
    "not guarantees of race outcomes."
)

# 4. Load dataset and model
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except FileNotFoundError as error:
    st.error(f"File not found: {error.filename}")
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

# 5. Features used during training
FEATURES = [
    "year",
    "round",
    "grid",
    "driver_prev_races",
    "driver_prev_avg_position",
    "driver_prev_avg_points",
    "driver_prev_5_avg_position",
    "driver_prev_wins",
    "driverId",
    "constructorName",
]

required_columns = FEATURES + [
    "race_name",
    "givenName",
    "familyName",
    "target_win",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

# 6. Select historical test races

test_df = df[df["year"] == 2025].copy()

if test_df.empty:
    st.error("No 2025 race records were found in the dataset.")
    st.stop()

test_df["race_label"] = (
    test_df["year"].astype(str)
    + " | Round "
    + test_df["round"].astype(str)
    + " | "
    + test_df["race_name"].astype(str)
)

race_options = (
    test_df[["race_label", "round"]]
    .drop_duplicates()
    .sort_values("round")["race_label"]
    .tolist()
)

selected_race = st.selectbox(
    "Select a 2025 Grand Prix",
    race_options
)

race_df = test_df[
    test_df["race_label"] == selected_race
].copy()

if race_df.empty:
    st.warning("No driver records were found for this race.")
    st.stop()

# 7. Predict driver scores
try:
    probabilities = model.predict_proba(
        race_df[FEATURES]
    )[:, 1]

    race_df["model_score"] = probabilities

except Exception as error:
    st.error(f"Could not generate predictions: {error}")
    st.stop()

race_df = race_df.sort_values(
    "model_score",
    ascending=False
).reset_index(drop=True)

race_df["rank"] = range(1, len(race_df) + 1)

race_df["driver"] = (
    race_df["givenName"].fillna("")
    + " "
    + race_df["familyName"].fillna("")
).str.strip()

# 8. Display the top-ranked driver
top_driver = race_df.iloc[0]

st.subheader("Model's Top-Ranked Driver")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Top-ranked driver",
        top_driver["driver"]
    )

with col2:
    st.metric(
        "Constructor",
        str(top_driver["constructorName"])
    )

with col3:
    st.metric(
        "Model score",
        f"{top_driver['model_score']:.1%}"
    )

st.caption(
    "The model score is not a guaranteed chance of winning. "
    "Scores for all drivers do not necessarily add up to 100%."
)

# 9. Compare with the actual result
actual_winners = race_df[race_df["target_win"] == 1]

if not actual_winners.empty:
    actual_winner = actual_winners.iloc[0]["driver"]

    st.subheader("Historical Result")

    if top_driver["target_win"] == 1:
        st.success(
            f"The model ranked {actual_winner} first, "
            "and that driver won this race."
        )
    else:
        st.warning(
            f"Actual winner: {actual_winner}. "
            f"The model ranked {top_driver['driver']} first."
        )

# 10. Display all driver rankings
st.subheader("Full Driver Rankings")

display_df = race_df[
    [
        "rank",
        "driver",
        "constructorName",
        "grid",
        "model_score",
        "target_win",
    ]
].copy()

display_df = display_df.rename(
    columns={
        "rank": "Rank",
        "driver": "Driver",
        "constructorName": "Constructor",
        "grid": "Starting Grid",
        "model_score": "Model Score",
        "target_win": "Actual Winner (1 = Yes)",
    }
)

display_df["Model Score"] = display_df[
    "Model Score"
].map(lambda score: f"{score:.1%}")

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)

# 11. Project information
with st.expander("About this project"):
    st.write(
        """
        **Project:** F1 Race Winner Predictor

        **Machine Learning algorithm:** Logistic Regression

        **Training data:** 2023 and 2024 seasons

        **Evaluation data:** 2025 season

        **Input features:** Starting grid, driver history,
        previous points, previous positions, driver identity,
        constructor and race details.

        **Evaluation metric:** Correct top-ranked driver per race.
        """
    )

st.caption(
    "Built with Python, pandas, scikit-learn and Streamlit."
)
