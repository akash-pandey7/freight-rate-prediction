import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

def main():
    # Use DATA_DIR to load the files
    train = pd.read_csv(DATA_DIR / "train-test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA_DIR / "december-chart-inputs.csv", parse_dates=["date"])
    template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")

    print("=" * 70)
    print("SHAPES")
    print("=" * 70)
    for name, df in [("train_test", train), ("validation", val),
                    ("december_chart_inputs", dec), ("template", template)]:
        print(f"{name:25s} {df.shape}")

    print("\n" + "=" * 70)
    print("TRAIN_TEST DTYPES + NULLS")
    print("=" * 70)
    print(train.dtypes) # Check Data types
    print("\nNull counts:")
    print(train.isna().sum()) # Check for nulls

    print("\n" + "=" * 70)
    print("DATE RANGE (train vs validation)")
    print("=" * 70)
    print("train_test date range:", train["date"].min(), "->", train["date"].max()) # Date range of train_test
    print("validation date range:", val["date"].min(), "->", val["date"].max()) # Date range of validation

    print("\n" + "=" * 70)
    print("TARGET (posted_rate) DISTRIBUTION")
    print("=" * 70)
    print(train["posted_rate"].describe()) # posted_rate distribution stats
    print("Potential outliers (posted_rate <= 0):", (train["posted_rate"] <= 0).sum())
    
    print("Top 10 highest posted_rate:")
    print(train.nlargest(10, "posted_rate")[["load_id", "distance", "equipment", "posted_rate"]])
    print("Bottom 10 lowest posted_rate:")
    print(train.nsmallest(10, "posted_rate")[["load_id", "distance", "equipment", "posted_rate"]])

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK: market_index / quote_signal vs posted_rate")
    print("=" * 70)
    corr = train[["market_index", "quote_signal", "posted_rate", "distance", "weight"]].corr() # Check correlation of posted_rate with market_index and quote_signal
    print(corr["posted_rate"].sort_values(ascending=False))

    print("\n" + "=" * 70)
    print("EQUIPMENT VALUE COUNTS")
    print("=" * 70)
    print(train["equipment"].value_counts())
    print("\nEquipment values NOT in train but in validation (if any):")
    print(set(val["equipment"].unique()) - set(train["equipment"].unique()))
    print("Equipment values in december_chart_inputs:", dec["equipment"].unique())

    print("\n" + "=" * 70)
    print("LANE / GEO SANITY CHECK")
    print("=" * 70)
    print("Unique pickup cities (train):", train["pickup"].nunique())
    print("Unique delivery cities (train):", train["delivery"].nunique())
    # Check if December's fixed lane (Lexington -> Fort Wayne) actually appears in train_test
    fixed_lane_mask = (train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")
    print(f"Lexington -> Fort Wayne appears {fixed_lane_mask.sum()} times in train_test")
    # If it appears 0 times, December predictions are a pure extrapolation test.

    print("\n" + "=" * 70)
    print("TEMPLATE / ID FORMAT CHECK")
    print("=" * 70)
    print(template.head())
    print("load_id format sample:", template["load_id"].iloc[0])
    print("Columns in december_chart_inputs:", dec.columns.tolist())
    print("Do template and validation.csv load_ids match exactly : ", set(template["load_id"]) == set(val["load_id"]))
    
    print("\n" + "=" * 70)
    print("WEIGHT SIGN CHECK")
    print("=" * 70)
    print("Rows with weight <= 0 (train):", (train["weight"] <= 0).sum())
    print("Rows with weight <= 0 (validation):", (val["weight"] <= 0).sum())
    print(train[train["weight"] <= 0][["load_id", "weight", "distance", "posted_rate"]].head())

if __name__ == "__main__":
    main()