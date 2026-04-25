import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

def prepare_matrices(seasonal_signatures, yield_df):
    """
    Aligns Prithvi embeddings with historical yield data.
    Keeps 'year' in the dataframe so we can split by time.
    """
    # Merge data
    data = seasonal_signatures.merge(yield_df, on=['year', 'state_fips', 'county'])
    
    # Extract year before we strip the dataframe for X and y
    years = data['year'].values
    
    # Feature Matrix X (the 768-dim embeddings)
    X = np.stack(data['embedding_vector'].values)
    
    # Target Vector y (Yield)
    y = data['yield_bu_acre'].values
    
    return X, y, years

def train_and_validate(X, y, years):
    """
    Performs a simple Time-Based Split (Baseline).
    Trains on years < 2023, Tests on 2023-2024.
    """
    # Define the split point
    test_mask = years >= 2023
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Optimized for speed and performance
    params = {
        'n_estimators': 200,      # Increased since we're only training once
        'max_depth': 6,           # Can afford slightly more depth with 9k records
        'learning_rate': 0.05,    # Slower learning rate for better stability
        'subsample': 0.8,
        'tree_method': 'hist',    # Significantly speeds up training
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    print(f"Starting Baseline Training...")
    print(f"Training records (Pre-2023): {len(X_train)}")
    print(f"Testing records (2023-2024): {len(X_test)}")

    # Initialize and fit the XGBoost Regressor once
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Predict on the test set
    all_preds = model.predict(X_test)
    all_actuals = y_test

    # Calculate MAE
    mae = mean_absolute_error(all_actuals, all_preds)
    print(f"Baseline Validation MAE: {mae:.2f} bu/acre")
    
    return mae, all_actuals, all_preds