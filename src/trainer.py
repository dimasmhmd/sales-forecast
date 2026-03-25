from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    # Tentukan jumlah CV secara dinamis
    # Jika data sangat sedikit, gunakan cv=2. Jika cukup, gunakan cv=3 atau lebih.
    n_samples = len(X)
    if n_samples < 10:
        cv_folds = 2
    elif n_samples < 30:
        cv_folds = 3
    else:
        cv_folds = 5

    param_grid = {
        'n_estimators': [100, 500], # Dikurangi sedikit agar tidak berat di Streamlit
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 0.9]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=10, # Kurangi iterasi untuk testing cepat
        cv=cv_folds, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
