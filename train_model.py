
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "f1_model_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load prepared data
df = pd.read_csv(DATA_FILE)

# Train on older seasons; test on 2025
train_df = df[df["year"].isin([2023, 2024])].copy()
test_df = df[df["year"] == 2025].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing data is empty. Check the CSV seasons.")

# Features available before the race result
numeric_features = [
    "year",
    "round",
    "grid",
    "driver_prev_races",
    "driver_prev_avg_position",
    "driver_prev_avg_points",
    "driver_prev_5_avg_position",
    "driver_prev_wins",
]

categorical_features = [
    "driverId",
    "constructorName",
]

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns]
y_train = train_df["target_win"]

X_test = test_df[feature_columns]
y_test = test_df["target_win"]


# Create a fresh preprocessing pipeline for each model
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


# Define both models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}


# Baseline: choose the driver starting in P1
pole_sitters = test_df[test_df["grid"] == 1].copy()

pole_sitter_wins = (
    pole_sitters.groupby(["year", "round"])["target_win"].max()
)

baseline_rate = pole_sitter_wins.mean() * 100

print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Pole-sitter baseline:", round(baseline_rate, 2), "%")


# Train, evaluate, compare and save both models
results = []
all_top_picks = {}

for model_name, model in models.items():
    print(f"\n{'=' * 12} {model_name} {'=' * 12}")
    print("Training...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nClassification metrics:")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4),
    )

    if y_test.nunique() == 2:
        print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    # Rank drivers within each race
    race_predictions = test_df[
        [
            "year", "round", "race_name",
            "driverId", "givenName", "familyName", "target_win",
        ]
    ].copy()

    race_predictions["win_probability"] = probabilities

    top_picks = race_predictions.loc[
        race_predictions.groupby(["year", "round"])[
            "win_probability"
        ].idxmax()
    ].copy()

    top_picks = top_picks.sort_values(["year", "round"])

    top1_rate = top_picks["target_win"].mean() * 100
    correct_picks = int(top_picks["target_win"].sum())

    print("\nRace-winner ranking:")
    print(f"Correct top picks: {correct_picks} out of {len(top_picks)}")
    print("Top-pick success rate:", round(top1_rate, 2), "%")

    print("\nAll race predictions:")
    print(
        top_picks[
            [
                "year", "round", "race_name",
                "givenName", "familyName",
                "win_probability", "target_win",
            ]
        ].to_string(index=False)
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Balanced accuracy": round(
            balanced_accuracy_score(y_test, predictions), 4
        ),
        "ROC-AUC": (
            round(roc_auc_score(y_test, probabilities), 4)
            if y_test.nunique() == 2 else None
        ),
        "Correct top picks": f"{correct_picks}/{len(top_picks)}",
        "Top-pick success (%)": round(top1_rate, 2),
    })

    all_top_picks[model_name] = top_picks

    # Save each complete pipeline separately
    filename = (
        "f1_logistic_model.joblib"
        if model_name == "Logistic Regression"
        else "f1_random_forest_model.joblib"
    )

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)
    print("\nModel saved to:", model_path)


# Final comparison
comparison = pd.DataFrame(results)

print("\n" + "=" * 12 + " FINAL MODEL COMPARISON " + "=" * 12)
print(comparison.to_string(index=False))
print("\nPole-sitter baseline:", round(baseline_rate, 2), "%")
print("Compare Top-pick success (%) with the baseline.")
