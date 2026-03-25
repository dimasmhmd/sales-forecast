from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    """Mencari setelan terbaik dan melatih model XGBoost"""
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # RandomizedSearch jauh lebih cepat untuk aplikasi web dibanding GridSearch
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_