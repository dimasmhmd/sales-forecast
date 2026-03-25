import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px # Tambahan untuk grafik importance
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Forecast Pro", layout="wide", page_icon="📈")

st.title("📊 Enterprise Sales Forecasting Dashboard")
st.markdown("""
Aplikasi ini menggunakan algoritma **XGBoost Teroptimasi** untuk memprediksi tren penjualan masa depan 
berdasarkan data historis, efek musiman, dan pola sebelumnya.
""")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV Penjualan", type="csv")
st.sidebar.info("Pastikan CSV memiliki kolom 'tanggal' (YYYY-MM-DD) dan 'sales'.")

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    
    # Cek Validitas Kolom
    if 'tanggal' in df.columns and 'sales' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df = df.set_index('tanggal').sort_index()
        
        # --- FITUR 1: TAMPILAN DATA MENTAH DENGAN SCROLL ---
        st.subheader("📋 Data Mentah (Preview)")
        st.markdown("Berikut adalah data yang Anda unggah. Gunakan scrollbar jika data terlalu panjang.")
        
        # Gunakan st.dataframe dengan height untuk mengaktifkan scrolling otomatis
        st.dataframe(df, use_container_width=True, height=300) 
        
        # --- TOMBOL FORECASTING ---
        st.sidebar.markdown("---")
        if st.sidebar.button("Mulai Forecasting & Analisis"):
            with st.spinner('Menganalisis pola dan mengoptimalkan model (ini mungkin memakan waktu)...'):
                
                # 2. Feature Engineering
                data_final = prepare_features(df)
                
                # Cek jika data cukup setelah diproses
                if len(data_final) < 10:
                    st.error("Data terlalu sedikit setelah diproses (minimal 40 baris data mentah dibutuhkan untuk Lag-30).")
                    st.stop()
                
                # Split Feature & Target
                X = data_final.drop(columns=['sales'])
                y = data_final['sales']
                
                # 3. Training & Tuning (Panggil fungsi yang baru)
                model, best_params = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # --- BAGIAN HASIL ---
                st.divider()
                st.subheader("🚀 Hasil Analisis & Prediksi")
                
                # Metriks Utama
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediksi Hari Besok", f"{int(preds[-1])} Unit")
                c2.metric("Tren (vs Kemarin)", f"{int(preds[-1] - y.iloc[-1])} Unit", delta_color="normal")
                c3.metric("Rekomendasi Stok Safety (120%)", f"{int(preds[-1] * 1.2)} Unit")

                # Grafik Tren (Plotly)
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=y.index[-60:], y=y[-60:], name="Data Aktual", line=dict(color='#1f77b4', width=2)))
                fig_trend.add_trace(go.Scatter(x=y.index[-60:], y=preds[-60:], name="Prediksi Model", line=dict(color='#ff7f0e', dash='dot')))
                
                fig_trend.update_layout(
                    title="Perbandingan 60 Hari Terakhir (Aktual vs Prediksi)",
                    xaxis_title="Tanggal",
                    yaxis_title="Total Penjualan",
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- FITUR 2: FEATURE IMPORTANCE ---
                st.divider()
                st.subheader("🧠 Faktor Berpengaruh (Feature Importance)")
                st.markdown("Grafik ini menunjukkan variabel mana yang paling dominan mempengaruhi prediksi model XGBoost.")

                # Ambil Importance dari Model
                importance_scores = model.feature_importances_
                feature_names = X.columns
                
                # Buat DataFrame Importance
                df_importance = pd.DataFrame({'Fitur': feature_names, 'Importance': importance_scores})
                df_importance = df_importance.sort_values(by='Importance', ascending=True) # Sort untuk grafik

                # Grafik Batang Horizontal (Plotly Express)
                fig_import = px.bar(df_importance, x='Importance', y='Fitur', orientation='h',
                                     title="Faktor Dominan dalam Prediksi Sales",
                                     labels={'Importance': 'Tingkat Kepentingan (0-1)'},
                                     color='Importance', color_continuous_scale='Reds')
                
                fig_import.update_layout(template="plotly_white", yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_import, use_container_width=True)

                st.success(f"Model berhasil dioptimalkan.")
                with st.expander("Lihat Parameter Terbaik Model"):
                    st.write(best_params)

    else:
        st.error("Format CSV salah! Pastikan ada kolom 'tanggal' dan 'sales'.")
else:
    # Tampilan Awal jika belum upload
    st.info("👋 Selamat datang! Silakan unggah data penjualan Anda (CSV) di sidebar untuk memulai analisis.")
    
    # Tampilkan contoh format data
    st.markdown("### Contoh Format CSV yang Benar:")
    example_data = pd.DataFrame({
        'tanggal': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'sales': [100, 120, 95]
    })
    st.dataframe(example_data)
