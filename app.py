import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Forecast Pro", layout="wide", page_icon="📈")

st.title("📊 Enterprise Sales Forecasting Dashboard")
st.markdown("""
Aplikasi ini menggunakan algoritma **XGBoost Teroptimasi** untuk memprediksi tren penjualan masa depan.
""")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV Penjualan", type="csv")

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    
    if 'tanggal' in df.columns and 'sales' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df = df.set_index('tanggal').sort_index()
        
        # --- FITUR: DATA MENTAH DALAM TOMBOL (EXPANDER) ---
        with st.expander("👁️ Klik untuk Melihat/Sembunyikan Data Mentah"):
            st.markdown("Gunakan scrollbar di dalam tabel jika data terlalu banyak.")
            # Tetap gunakan height agar bisa di-scroll di dalam expander
            st.dataframe(df, use_container_width=True, height=400) 
        
        # --- TOMBOL FORECASTING ---
        st.sidebar.markdown("---")
        if st.sidebar.button("Mulai Forecasting & Analisis"):
            with st.spinner('Menganalisis pola dan mengoptimalkan model...'):
                
                # 2. Feature Engineering
                data_final = prepare_features(df)
                
                if len(data_final) < 15:
                    st.error("Data terlalu sedikit setelah diproses (minimal butuh ~45 baris data mentah).")
                    st.stop()
                
                # Split Feature & Target
                X = data_final.drop(columns=['sales'])
                y = data_final['sales']
                
                # 3. Training & Tuning
                model, best_params = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # --- HASIL PREDIKSI ---
                st.divider()
                st.subheader("🚀 Hasil Analisis & Prediksi")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediksi Hari Besok", f"{int(preds[-1])} Unit")
                c2.metric("Tren (vs Kemarin)", f"{int(preds[-1] - y.iloc[-1])} Unit", delta_color="normal")
                c3.metric("Rekomendasi Stok Safety", f"{int(preds[-1] * 1.2)} Unit")

                # Grafik Tren
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=y.index[-60:], y=y[-60:], name="Data Aktual", line=dict(color='#1f77b4', width=2)))
                fig_trend.add_trace(go.Scatter(x=y.index[-60:], y=preds[-60:], name="Prediksi Model", line=dict(color='#ff7f0e', dash='dot')))
                fig_trend.update_layout(title="Tren 60 Hari Terakhir", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- FITUR: FEATURE IMPORTANCE ---
                st.divider()
                st.subheader("🧠 Faktor Berpengaruh (Feature Importance)")
                
                importance_scores = model.feature_importances_
                df_importance = pd.DataFrame({'Fitur': X.columns, 'Importance': importance_scores})
                df_importance = df_importance.sort_values(by='Importance', ascending=True)

                fig_import = px.bar(df_importance, x='Importance', y='Fitur', orientation='h',
                                     color='Importance', color_continuous_scale='Reds',
                                     labels={'Importance': 'Skor Kepentingan'})
                fig_import.update_layout(template="plotly_white")
                st.plotly_chart(fig_import, use_container_width=True)

                st.success("Analisis Selesai!")
    else:
        st.error("Format CSV salah! Pastikan ada kolom 'tanggal' and 'sales'.")
else:
    st.info("👋 Silakan unggah data penjualan Anda (CSV) untuk memulai.")
