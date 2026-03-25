import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_col='Units Sold'):
    """Fungsi ini mengolah data mentah menjadi fitur siap pakai oleh AI"""
    df = df.copy()
    
    # Otomatis cari kolom tanggal
    date_cols = [c for c in df.columns if 'Date' in c]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.set_index(date_cols[0]).sort_index()

    # Fitur Waktu
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

    # Label Encoding untuk kolom teks (kategorikal)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Lag Features
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()
