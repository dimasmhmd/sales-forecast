import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df, target_col='Units Sold'):
    """Memproses data mentah menjadi fitur untuk Machine Learning"""
    df = df.copy()
    
    # 1. Identifikasi & Set Tanggal sebagai Index
    date_cols = [c for c in df.columns if 'Date' in c]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

    # 2. Fitur Waktu (Temporal)
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year

    # 3. Label Encoding untuk Kolom Teks (Kategorikal)
    # Ini mengubah 'Cosmetics' -> 0, 'Fruits' -> 1, dll.
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Lag Features (Data masa lalu)
    # Model belajar dari performa 1 hari dan 7 hari sebelumnya
    for lag in [1, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Hapus baris kosong akibat shifting (lag)
    return df.dropna()
