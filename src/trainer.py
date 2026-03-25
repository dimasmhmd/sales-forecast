from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    """Melatih model XGBoost dengan optimasi parameter otomatis"""
    
    # Tentukan jumlah Cross-Validation secara dinamis
    n_samples = len(X)
    if n_samples < 15:
        cv_folds = 2
    elif n_samples < 50:
        cv_folds = 3
    else:
        cv_folds = 5

    # Parameter yang akan diuji
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Mencari kombinasi terbaik dari 5 percobaan acak
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=5, 
        cv=cv_folds, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    
    return search.best_estimator_, search.best_params_
