import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_col='Units Sold'):
    df = df.copy()
    
    # 1. Pastikan Tanggal adalah Index
    date_cols = [c for c in df.columns if 'Date' in c or c == 'tanggal']
    if date_cols:
        col = date_cols[0]
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col).sort_index()

    # 2. Fitur Waktu
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year

    # 3. Label Encoding (Hanya jika ada kolom teks sisa)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Lag Features
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()
