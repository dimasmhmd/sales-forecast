import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_col='Units Sold'):
    df = df.copy()
    
    # 1. Konversi Tanggal Otomatis (Cari kolom yang mengandung kata 'Date')
    date_cols = [c for c in df.columns if 'Date' in c]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.set_index(date_cols[0]).sort_index()

    # 2. Fitur Waktu
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # 3. Handle Kolom Teks (Kategorikal)
    # Otomatis deteksi kolom bertipe object/string
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Lag Features untuk Target
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()
