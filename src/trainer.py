from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    """Mencari setelan XGBoost paling akurat secara cepat"""
    n_samples = len(X)
    cv_folds = 2 if n_samples < 40 else 3

    # Parameter grid yang seimbang antara kecepatan & akurasi
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # RandomizedSearch agar aplikasi tidak hang (mencoba 3 kombinasi acak)
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=3, 
        cv=cv_folds, 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
