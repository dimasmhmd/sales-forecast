import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Analytics Pro", layout="wide", page_icon="📊")

st.title("🌎 Global Sales Forecasting & Analytics")
st.markdown("""
Dashboard ini mampu memproses dataset penjualan multivariat. Gunakan filter di sidebar untuk menyaring data berdasarkan kategori tertentu.
""")

# --- SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📂 Data & Filter")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    
    # Identifikasi kolom tanggal otomatis
    date_col = [c for c in df.columns if 'Date' in c]
    if not date_col:
        st.error("Gagal menemukan kolom tanggal (harus mengandung kata 'Date').")
        st.stop()
    
    date_col = date_col[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # --- FITUR FILTER DINAMIS ---
    st.sidebar.subheader("🎯 Filter Analisis")
    
    # Pilih Region (Jika ada kolom Region)
    selected_region = "All"
    if 'Region' in df.columns:
        regions = ["All"] + sorted(df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Pilih Wilayah (Region)", regions)

    # Pilih Item Type (Jika ada kolom Item Type)
    selected_item = "All"
    if 'Item Type' in df.columns:
        items = ["All"] + sorted(df['Item Type'].unique().tolist())
        selected_item = st.sidebar.selectbox("Pilih Jenis Barang", items)

    # Terapkan Filter ke DataFrame
    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_item != "All":
        filtered_df = filtered_df[filtered_df['Item Type'] == selected_item]

    # Pilih Target Prediksi (Kolom Angka)
    target_options = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    default_target = 'Units Sold' if 'Units Sold' in target_options else target_options[0]
    selected_target = st.sidebar.selectbox("Target Prediksi", target_options, index=target_options.index(default_target))

    # --- TAMPILAN DATA ---
    with st.expander("👁️ Lihat Data Terfilter"):
        st.write(f"Menampilkan {len(filtered_df)} baris data.")
        st.dataframe(filtered_df, use_container_width=True, height=300)

    # --- PROSES FORECASTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Jalankan Forecasting"):
        if len(filtered_df) < 20:
            st.warning("⚠️ Data hasil filter terlalu sedikit (minimal 20 baris) untuk membuat prediksi yang akurat.")
        else:
            with st.spinner(f'Menganalisis tren {selected_target}...'):
                
                # 2. Feature Engineering (Gunakan filtered_df)
                # Pastikan src/processor.py sudah mendukung Label Encoding seperti yang dijelaskan sebelumnya
                data_final = prepare_features(filtered_df, target_col=selected_target)
                
                X = data_final.drop(columns=[selected_target])
                y = data_final[selected_target]
                
                # 3. Training
                model, best_params = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # --- DASHBOARD HASIL ---
                st.divider()
                st.subheader(f"🚀 Analisis Prediksi: {selected_item} di {selected_region}")
                
                # Metriks
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediksi Berikutnya", f"{preds[-1]:,.0f}")
                m2.metric("Total Data Terolah", len(y))
                m3.metric("Rekomendasi Stok", f"{(preds[-1] * 1.15):,.0f}")

                # Grafik Visualisasi
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index, y=y, name="Aktual", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi Model", line=dict(color='#ff7f0e', dash='dot')))
                fig.update_layout(title=f"Tren Historis vs Model ({selected_target})", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # --- FEATURE IMPORTANCE ---
                st.subheader("🧠 Faktor Penentu Penjualan")
                importance = pd.DataFrame({'Fitur': X.columns, 'Importance': model.feature_importances_})
                importance = importance.sort_values(by='Importance', ascending=True).tail(10)
                
                fig_imp = px.bar(importance, x='Importance', y='Fitur', orientation='h', 
                                 title="10 Faktor Paling Berpengaruh", color='Importance',
                                 color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)

                st.success("Analisis selesai!")
else:
    st.info("👋 Silakan unggah file CSV global sales Anda untuk memulai.")
