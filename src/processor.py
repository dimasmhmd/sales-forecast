import pandas as pd

def prepare_features(df, target_col='sales'):
    """Menambahkan Fitur Lag, Rolling Mean, dan Time-based secara otomatis"""
    df = df.copy()
    
    # Pastikan index adalah datetime
    df.index = pd.to_datetime(df.index)
    
    # 1. Fitur Waktu (Temporal Features)
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    
    # 2. Lag Features (Melihat ke belakang)
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # 3. Rolling Window (Tren rata-rata seminggu)
    df['rolling_mean_7'] = df[target_col].shift(1).rolling(window=7).mean()
    
    # Hapus baris yang memiliki nilai kosong akibat shifting
    return df.dropna()