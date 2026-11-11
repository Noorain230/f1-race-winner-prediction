#  F1 Race Winner Predictor

A Machine Learning project that predicts potential Formula 1 race winners using historical race data and driver performance.

## About the Project

This project analyzes Formula 1 race results from the 2023–2025 seasons to estimate which driver is most likely to win a race.

It includes data collection, preprocessing, feature engineering, model training, and an interactive Streamlit application.

## Features

- Historical F1 data collection using FastF1.
- Driver performance analysis.
- Data preprocessing and feature engineering.
- Race winner prediction using Machine Learning.
- Model evaluation and comparison.
- Interactive Streamlit interface.

## Tech Stack

- **Language:** Python
- **Data Collection:** FastF1
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Web Application:** Streamlit

## Machine Learning Models

- Logistic Regression
- Random Forest Classifier

Models are trained on 2023–2024 data and evaluated on the 2025 season.

## Model Evaluation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 94.99% | 95.20% |
| ROC-AUC | 0.9651 | 0.9614 |
| Race-level top-pick success | 62.50% | 37.50% |

*Note: The dataset is imbalanced, and accuracy alone does not reflect winner-class prediction performance. Both models have limitations at the default classification threshold.*

## Run Locally

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd f1-race-winner-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Incorporate additional driver and constructor statistics.
- Improve winner prediction performance.
- Experiment with additional ML algorithms.
- Expand historical race data.

## Disclaimer

This project is for educational purposes. Predictions are estimates and do not guarantee actual race outcomes. Formula 1 trademarks belong to their respective owners.

## License

This project is licensed under the MIT License.
