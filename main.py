import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

DATA_DIR = Path('data')

# 1. Load Data
train = pd.read_csv(DATA_DIR / "train-test.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")
dec = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

# 2. Feature Engineering
# The December dataset only includes basic features, so I restricted training 
# to features present across all datasets to ensure consistency.
# market_index and quoted_signal were checked for correlation with posted rate (~0.03 and -0.04 respectively)
# too weak to be worth added to the model. Additionally, they are not present in the December dataset, so they were dropped everywhere from training.
def prepare_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    return df

FEATURE_COLS = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'month', 'dayofweek', 'day']
cat_cols = ['pickup', 'delivery', 'equipment'] # Categorical columns to be one-hot encoded
num_cols = ['distance', 'weight', 'month', 'dayofweek', 'day'] # Numerical columns to be standardized

train = prepare_features(train)

# 3. Time-based validation split
# The train-test.csv dataset is already sorted by date, so we can split it into 80% fit
# and 20% validation based on the date order.
# This ensures that the model is trained on past data and validated on future data.
assert train['date'].is_monotonic_increasing, "train-test.csv is not date-sorted — sort before splitting"
cutoff = int(len(train) * 0.8)
fit_df, train_val_df = train.iloc[:cutoff], train.iloc[cutoff:]
print(f"Fit: {fit_df.min()['date']} -> {fit_df.max()['date']}, {len(fit_df)} rows")
print(f"Validation: {train_val_df.min()['date']} -> {train_val_df.max()['date']}, {len(train_val_df)} rows")

# Standardize numerical features and one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
# Define the model pipeline with preprocessing and a Random Forest regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
])

# 4. Fit model on 80% of the train-test data, validate on the remaining 20%
model.fit(fit_df[FEATURE_COLS], fit_df['posted_rate'])
train_val_preds = model.predict(train_val_df[FEATURE_COLS])

metrics = {
    'mae' : mean_absolute_error(train_val_df['posted_rate'], train_val_preds),
    'mape' : mean_absolute_percentage_error(train_val_df['posted_rate'], train_val_preds),
    'r2' : r2_score(train_val_df['posted_rate'], train_val_preds)
}
print("Validation Metrics:", metrics)
Path('artifacts').mkdir(exist_ok=True)
with open('artifacts/validation_metrics.txt', 'w') as f:
    json.dump(metrics, f)

# 5. Refit on full train-test data for final model
model.fit(train[FEATURE_COLS], train['posted_rate']) 

# 6. Generate Validation Output (Exactly 2 columns: load_id, predicted_rate)
X_val = prepare_features(val)[FEATURE_COLS]
val_out = pd.DataFrame({
    'load_id': val['load_id'],
    'predicted_rate': model.predict(X_val).round(2)
})
val_out.to_csv('validation_predictions.csv', index=False)

# 7. Generate December Chart Output
X_dec = prepare_features(dec)[FEATURE_COLS]
dec_out = dec.copy()
dec_out['predicted_rate'] = model.predict(X_dec).round(2)
dec_out = dec_out[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]
dec_out.to_csv('december_chart_outputs.csv', index=False)

print("Pipeline complete. Output files generated.")