import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# Konfigurasi Halaman
st.set_page_config(page_title="Sales Forecast Pro", layout="wide", page_icon="📈")

st.title("📊 Enterprise Sales Forecasting Dashboard")
st.markdown("""
Aplikasi ini menggunakan algoritma **XGBoost Teroptimasi** untuk memprediksi tren penjualan masa depan 
berdasarkan data historis, efek musiman, dan pola promosi.
""")

# --- SIDEBAR ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV Penjualan", type="csv")
st.sidebar.info("Pastikan CSV memiliki kolom 'tanggal' (YYYY-MM-DD) dan 'sales'.")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    if 'tanggal' in df.columns and 'sales' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df = df.set_index('tanggal').sort_index()
        
        # Tampilkan Data Sekilas
        with st.expander("Lihat Data Mentah"):
            st.write(df.tail(10))

        # --- PROSES MODEL ---
        if st.sidebar.button("Mulai Forecasting"):
            with st.spinner('Menganalisis pola dan mengoptimalkan model...'):
                # Feature Engineering
                data_final = prepare_features(df)
                
                # Split Feature & Target
                X = data_final.drop(columns=['sales'])
                y = data_final['sales']
                
                # Training dengan Tuning
                model, best_params = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # --- METRIKS UTAMA ---
                st.subheader("🚀 Hasil Analisis")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediksi Besok", f"{int(preds[-1])} Unit")
                c2.metric("Tren (vs Kemarin)", f"{int(preds[-1] - y.iloc[-1])} Unit", delta_color="normal")
                c3.metric("Rekomendasi Stok Safety", f"{int(preds[-1] * 1.2)} Unit")

                # --- VISUALISASI ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index[-60:], y=y[-60:], name="Data Aktual", line=dict(color='#1f77b4', width=2)))
                fig.add_trace(go.Scatter(x=y.index[-60:], y=preds[-60:], name="Prediksi Model", line=dict(color='#ff7f0e', dash='dot')))
                
                fig.update_layout(
                    title="Perbandingan 60 Hari Terakhir (Aktual vs Prediksi)",
                    xaxis_title="Tanggal",
                    yaxis_title="Total Penjualan",
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"Model dioptimalkan dengan setelan: {best_params}")
    else:
        st.error("Format CSV salah! Pastikan ada kolom 'tanggal' dan 'sales'.")
else:
    st.info("👋 Selamat datang! Silakan unggah data penjualan Anda di sidebar untuk memulai analisis.")