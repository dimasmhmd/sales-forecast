from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    n_samples = len(X)
    cv_folds = 2 if n_samples < 40 else 3

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
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
