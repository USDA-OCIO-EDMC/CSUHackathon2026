import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

def prepare_matrices(seasonal_signatures, yield_df):
    """
    Aligns Prithvi embeddings with historical yield data and 
    converts them into a format compatible with XGBoost.
    """
    # Merge growth signatures with ground-truth yield based on spatio-temporal keys
    data = seasonal_signatures.merge(yield_df, on=['year', 'state_fips', 'county'])
    
    # Flatten list of embeddings into a 2D NumPy array (Feature Matrix X)
    # Target shape: (n_samples, n_features)
    X = np.stack(data['embedding_vector'].values)
    
    # Target vector (Yield values)
    y = data['yield_bu_acre'].values
    
    return X, y

def train_and_validate(X, y):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) to assess model 
    generalization on small historical datasets (n=~20 per county).
    """
    loo = LeaveOneOut()
    all_preds = []
    all_actuals = []

    # Default parameters optimized for tabular geospatial features
    # max_depth is kept low to mitigate overfitting on short time-series
    params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    print("Starting LOOCV validation (simulating unseen years)...")

    for train_idx, test_idx in loo.split(X):
        # Split data: 19 years for training, 1 year for validation
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and fit the XGBoost Regressor
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Record predictions to evaluate global model variance
        pred = model.predict(X_test)
        all_preds.append(pred[0])
        all_actuals.append(y_test[0])

    # Calculate Mean Absolute Error (MAE) as the primary performance metric
    mae = mean_absolute_error(all_actuals, all_preds)
    print(f"Validation MAE: {mae:.2f} bu/acre")
    
    return mae, all_actuals, all_preds