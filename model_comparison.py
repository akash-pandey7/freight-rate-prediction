"""
model_comparison.py — Reproduces the RandomForest vs HistGradientBoosting
comparison referenced in the report's Model Selection section. Not part of
the main pipeline (see main.py for the final chosen model)

"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.model_selection import KFold

DATA_DIR = Path("data")

train = pd.read_csv(DATA_DIR / "train-test.csv")


def prepare_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["weight"] = df["weight"].abs()
    return df


FEATURE_COLS = ["pickup", "delivery", "distance", "equipment", "weight", "month", "dayofweek", "day"]
cat_cols = ["pickup", "delivery", "equipment"]
te_cols = ["pickup", "delivery"]
ohe_cols = ["equipment"]
num_cols = ["distance", "weight", "month", "dayofweek", "day"]

train = prepare_features(train)
assert train["date"].is_monotonic_increasing

cutoff = int(len(train) * 0.8)
fit_df, holdout_df = train.iloc[:cutoff], train.iloc[cutoff:]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "RandomForest (one-hot)": Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])),
        ("regressor", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
    ]),
    "HistGBM (target-encoded)": Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[
            ("num", "passthrough", num_cols),
            ('te', TargetEncoder(target_type='continuous', cv=kf), te_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ohe_cols),
        ])),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=500, max_depth=8, learning_rate=0.05,
            early_stopping=True, random_state=42, l2_regularization=0.1,
        )),
    ]),
}

print(f"{'Model':<26} {'MAE':>10} {'MAPE':>8} {'R2':>8}")
print("-" * 54)
for name, model in candidates.items():
    model.fit(fit_df[FEATURE_COLS], fit_df["posted_rate"])
    preds = model.predict(holdout_df[FEATURE_COLS])
    mae = mean_absolute_error(holdout_df["posted_rate"], preds)
    mape = mean_absolute_percentage_error(holdout_df["posted_rate"], preds)
    r2 = r2_score(holdout_df["posted_rate"], preds)
    print(f"{name:<26} {mae:>10.2f} {mape:>7.2%} {r2:>8.3f}")