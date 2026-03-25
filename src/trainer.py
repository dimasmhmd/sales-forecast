from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_optimized_xgb(X, y):
    """Mencari setelan terbaik dan melatih model XGBoost"""
    
    # Tentukan jumlah CV secara dinamis berdasarkan jumlah data
    n_samples = len(X)
    if n_samples < 15: # Lag 30 + CV minimal
        cv_folds = 2
    elif n_samples < 50:
        cv_folds = 3
    else:
        cv_folds = 5

    # Grid parameter yang disederhanakan agar cepat di Streamlit
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    # RandomizedSearch untuk efisiensi waktu
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=5, # Coba 5 kombinasi acak
        cv=cv_folds, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    
    # Mengambil model terbaik yang sudah dilatih
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    return best_model, best_params
