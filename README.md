# 📈 Sales Forecast Pro (XGBoost)

Aplikasi peramalan penjualan berbasis Machine Learning yang menggunakan **XGBoost** dengan **Hyperparameter Tuning** otomatis.

## ✨ Fitur Utama
- **Automated Feature Engineering**: Membuat fitur lag (H-1, H-7, H-30) dan rolling mean secara otomatis.
- **Smart Tuning**: Menggunakan `RandomizedSearchCV` untuk menemukan konfigurasi model terbaik.
- **Interactive Dashboard**: Visualisasi data aktual vs prediksi menggunakan Plotly.
- **Stock Recommendation**: Memberikan estimasi stok aman berdasarkan hasil prediksi.

## 🚀 Cara Menjalankan Lokal
1. Clone repository ini.
2. Instal library: `pip install -r requirements.txt`.
3. Jalankan aplikasi: `streamlit run app.py`.

## 🛠️ Tech Stack
- Python, Streamlit, XGBoost, Scikit-Learn, Plotly.