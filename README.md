# Freight Rate Prediction

Predicts freight `posted_rate` from load features (lane, distance, weight, equipment, date)
for the Spotter AI take-home assessment. Produces final predictions for `validation.csv`
and a 31-day fixed-lane forecast for December 2025.

## Repo structure

```
├── main.py           # Full training + prediction pipeline (run this)
├── EDA.ipynb          # Interactive exploratory analysis (data profiling & tabular outputs)
├── EDA.py                # Exploratory analysis (run first, informs pipeline.py)
├── score.py               # Provided by Spotter - validates output format, builds December chart
├── requirements.txt        # Dependencies for pipeline.py + EDA.py + score.py
├── data/                  # Place the provided CSVs here (not committed - see below)
├── validation_predictions.csv     # Output: final predictions for validation.csv
├── december_chart_outputs.csv     # Output: filled December predictions
```

## Setup

```bash
git clone https://github.com/akash-pandey7/freight-rate-prediction.git
cd freight-rate-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv/Scripts/activate
pip install -r requirements.txt
```

## Data

This repo doesn't include the raw data provided by Spotter. Place the four files
you were given into `data/`:

```
data/
├── train-test.csv
├── validation.csv
├── december-chart-inputs.csv
└── validation-predictions-template.csv
```

## Running

**1. Explore the data (optional, informs the decisions in `main.py`):**

```bash
python EDA.py
```

**2. Train the model and generate predictions:**

```bash
python main.py
```

This trains on a time-based train_val_df split, prints validation metrics (MAE / MAPE / R-squared)
to the console and to `artifacts/validation_metrics.txt`, refits on the full dataset,
then writes `validation_predictions.csv` and `december_chart_outputs.csv`.

**3. Validate the output format and generate the December chart:**

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_chart_outputs.csv
```

This produces `scorer_results/candidate_december.png`, included in the report.

## Approach summary

- **Split**: time-based, not random. `validation.csv`'s dates (Nov–Dec 2025) come
  strictly after `train-test.csv`'s (Jan–Oct 2025), so the train_val_df uses the last 20%
  of `train-test.csv` by date to mirror that same forward-looking setup.
- **Features**: `pickup`, `delivery`, `distance`, `equipment`, `weight`, plus `month`,
  `day of week`, and `day` extracted from `date`.
- **Dropped features**: `market_index` and `quote_signal` - correlation with
  `posted_rate` is ~0.03 and -0.04 respectively (too weak to be worth using), and
  neither column exists in `december-chart-inputs.csv`, so excluding them everywhere
  keeps one consistent feature set across validation and December.
- **Model**: `RandomForestRegressor` in a `scikit-learn` pipeline (`StandardScaler` +
  `OneHotEncoder` for `pickup`/`delivery`/`equipment`).
- **Results** (holdout) : MAE $171.79, MAPE 7.65%, R-Squared 0.79.

## Notes

- `The December 1st Spike` The spike is caused by a genuine outlier in the training data - a Lexington-origin load on June 1 priced at $3,631 (vs. the typical $750–950 for similar loads) - which the model generalizes from since it shares the same pickup city, distance, and weight profile, producing a inflated Dec 1-2 prediction before settling into its normal $842-852 range for the rest of the month.
- `market_index`/`quote_signal` were checked for leakage (not just weak correlation)
  before being dropped - see `EDA.py` output.