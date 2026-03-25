import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_col='Units Sold'):
    """Transformasi data mentah menjadi dataset Time-Series"""
    df = df.copy()
    
    # 1. Pastikan kolom tanggal menjadi Index
    # Mencari kolom tanggal secara dinamis
    date_cols = [c for c in df.columns if 'Date' in c or c == 'tanggal' or c == 'Tanggal']
    if date_cols:
        col = date_cols[0]
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col).sort_index()

    # 2. Fitur Kalender
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 3. Label Encoding (Jika masih ada kolom teks seperti Sales Channel)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Fitur Lag (H-1 dan H-7)
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()
