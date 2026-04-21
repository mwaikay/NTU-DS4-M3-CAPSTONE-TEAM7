"""
HDB Resale Price Prediction
============================
Uses LightGBM to predict Singapore HDB resale prices.

Usage:
    python predict.py --train train.csv --test test.csv --output submission.csv

Requirements:
    pip install lightgbm scikit-learn pandas numpy
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ── Constants ──────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "town", "flat_type", "flat_model", "full_flat_type",
    "planning_area", "mrt_name", "pri_sch_name", "sec_sch_name",
    "bus_stop_name", "street_name",
]

BINARY_FLAG_COLS = [
    "residential", "commercial", "market_hawker",
    "multistorey_carpark", "precinct_pavilion",
]

MALL_HAWKER_COUNT_COLS = [
    "Mall_Within_500m", "Mall_Within_1km", "Mall_Within_2km",
    "Hawker_Within_500m", "Hawker_Within_1km", "Hawker_Within_2km",
]

# Columns to drop before training (IDs, raw text, redundant coords, target)
DROP_COLS = [
    "id", "Tranc_YearMonth", "block", "address", "postal",
    "storey_range", "lower", "upper", "mid",
    "resale_price",
    "mrt_latitude", "mrt_longitude",
    "bus_stop_latitude", "bus_stop_longitude",
    "pri_sch_latitude", "pri_sch_longitude",
    "sec_sch_latitude", "sec_sch_longitude",
]

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.05,
    "num_leaves":        255,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "verbose":           -1,
    "n_jobs":            -1,
}


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a raw dataframe.
    Safe to call on both train and test sets.
    """
    df = df.copy()

    # Derived numeric features
    df["lease_remain"]        = 99 - df["hdb_age"]
    df["area_per_dwelling"]   = df["floor_area_sqm"] / (df["total_dwelling_units"] + 1)
    df["total_hawker_stalls"] = df["hawker_food_stalls"] + df["hawker_market_stalls"]

    # Binary Y/N flags → 0/1
    for col in BINARY_FLAG_COLS:
        df[col] = (df[col] == "Y").astype(int)

    # Count NaNs mean "none nearby" → fill with 0
    for col in MALL_HAWKER_COUNT_COLS:
        df[col] = df[col].fillna(0)

    # Distance NaNs → fill with median
    df["Mall_Nearest_Distance"] = df["Mall_Nearest_Distance"].fillna(
        df["Mall_Nearest_Distance"].median()
    )

    # Ensure date columns are integers
    df["Tranc_Year"]  = df["Tranc_Year"].astype(int)
    df["Tranc_Month"] = df["Tranc_Month"].astype(int)

    # Categorical encoding (LightGBM handles these natively)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    """Load, engineer, and return train/test splits ready for modelling."""
    print(f"Loading training data from: {train_path}")
    train = pd.read_csv(train_path, low_memory=False)

    print(f"Loading test data from:     {test_path}")
    test  = pd.read_csv(test_path,  low_memory=False)

    print(f"  Train shape: {train.shape}")
    print(f"  Test shape:  {test.shape}")

    train = engineer_features(train)
    test  = engineer_features(test)

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    cat_features = [c for c in feature_cols if train[c].dtype.name == "category"]

    print(f"\n  Total features : {len(feature_cols)}")
    print(f"  Categoricals   : {len(cat_features)}")

    X      = train[feature_cols]
    y      = train["resale_price"]
    X_test = test[feature_cols]

    return X, y, X_test, test["id"], cat_features


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list,
    val_size: float = 0.15,
    n_rounds: int = 1500,
    early_stopping: int = 50,
    log_every: int = 100,
):
    """
    Train a LightGBM model with a hold-out validation split.

    Returns:
        model      – trained LightGBM Booster
        val_rmse   – RMSE on the hold-out set
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42
    )

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  categorical_feature=cat_features, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, free_raw_data=False)

    print(f"\nTraining LightGBM  (train={len(X_tr):,}  val={len(X_val):,}) …")

    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(log_every),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"\n  Best iteration : {model.best_iteration}")
    print(f"  Validation RMSE: {val_rmse:>12,.2f}")

    return model, val_rmse


# ── Prediction & Submission ────────────────────────────────────────────────────

def make_submission(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    output_path: str,
):
    """Generate predictions and save a submission CSV."""
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    sub   = pd.DataFrame({"Id": test_ids, "resale_price": preds})
    sub.to_csv(output_path, index=False)

    print(f"\nSubmission saved → {output_path}")
    print(sub.head(10).to_string(index=False))
    return sub


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HDB Resale Price Prediction")
    parser.add_argument("--train",  default="train.csv",       help="Path to training CSV")
    parser.add_argument("--test",   default="test.csv",        help="Path to test CSV")
    parser.add_argument("--output", default="submission.csv",  help="Output submission CSV path")
    parser.add_argument("--val-size",   type=float, default=0.15,  help="Validation split fraction")
    parser.add_argument("--n-rounds",   type=int,   default=1500,  help="Max boosting rounds")
    parser.add_argument("--early-stop", type=int,   default=50,    help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load & engineer
    X, y, X_test, test_ids, cat_features = load_data(args.train, args.test)

    # Train
    model, val_rmse = train_model(
        X, y, cat_features,
        val_size=args.val_size,
        n_rounds=args.n_rounds,
        early_stopping=args.early_stop,
    )

    # Feature importance (top 20)
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=X.columns,
    ).sort_values(ascending=False)

    print("\nTop 20 features by gain:")
    print(importance.head(20).to_string())

    # Save submission
    make_submission(model, X_test, test_ids, args.output)
    print(f"\nDone. Validation RMSE: {val_rmse:,.2f}")


if __name__ == "__main__":
    main()
