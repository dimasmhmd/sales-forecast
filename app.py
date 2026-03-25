import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Analytics Pro", layout="wide", page_icon="📈")

st.title("🌎 Global Sales Forecasting Dashboard")
st.markdown("Analisis prediktif multivariat dengan **Loading Bar** & **Breakdown Analysis**.")

# --- 2. SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📁 Manajemen Data")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Deteksi Kolom Tanggal
    date_cols = [c for c in df.columns if 'Date' in c or 'tanggal' in c.lower()]
    if not date_cols:
        st.error("❌ Tidak ditemukan kolom tanggal.")
        st.stop()
    
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter Sidebar
    st.sidebar.subheader("🎯 Filter Analisis")
    selected_region = "All"
    if 'Region' in df.columns:
        regions = ["All"] + sorted(df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Wilayah (Region)", regions)

    selected_item = "All"
    if 'Item Type' in df.columns:
        items = ["All"] + sorted(df['Item Type'].unique().tolist())
        selected_item = st.sidebar.selectbox("Jenis Barang", items)

    view_mode = "Total Agregat"
    if selected_item == "All":
        view_mode = st.sidebar.radio("Mode Tampilan Forecast:", ["Total Agregat", "Breakdown per Item"])

    target_options = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    selected_target = st.sidebar.selectbox("Pilih Target Prediksi", target_options, index=0)

    f_df = df.copy()
    if selected_region != "All":
        f_df = f_df[f_df['Region'] == selected_region]

    # Preview Data
    with st.expander("👁️ Preview Data Terfilter"):
        preview_df = f_df.copy()
        if selected_item != "All":
            preview_df = preview_df[preview_df['Item Type'] == selected_item]
        st.dataframe(preview_df, use_container_width=True, height=250)

    # --- 3. PROSES FORECASTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Jalankan Forecasting"):
        
        if view_mode == "Total Agregat":
            if selected_item != "All":
                f_df = f_df[f_df['Item Type'] == selected_item]
            
            f_df_agg = f_df.groupby(date_col)[selected_target].sum().reset_index()
            
            if len(f_df_agg) < 20:
                st.warning("⚠️ Data terlalu sedikit untuk analisis.")
            else:
                # PROGRESS BAR UNTUK SINGLE FORECAST
                progress_text = "Memulai optimasi AI..."
                my_bar = st.progress(0, text=progress_text)
                
                # Tahap 1: Persiapan Fitur
                my_bar.progress(30, text="Melakukan Feature Engineering...")
                data_final = prepare_features(f_df_agg, target_col=selected_target)
                X, y = data_final.drop(columns=[selected_target]), data_final[selected_target]
                
                # Tahap 2: Training
                my_bar.progress(60, text="Mencari Hyperparameter terbaik (XGBoost)...")
                model, _ = train_optimized_xgb(X, y)
                
                # Tahap 3: Prediksi
                my_bar.progress(90, text="Menghasilkan angka prediksi...")
                preds = model.predict(X)
                
                time.sleep(0.5)
                my_bar.empty() # Hapus progress bar setelah selesai

                # --- DASHBOARD HASIL ---
                st.divider()
                st.subheader(f"📊 Forecast: {selected_item} ({selected_region})")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediksi Berikutnya", f"{preds[-1]:,.0f}")
                c2.metric("Total Sampel", len(y))
                c3.metric("Rekomendasi Stok", f"{(preds[-1] * 1.2):,.0f}")

                # Grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index, y=y, name="Aktual", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi", line=dict(color='#ff7f0e', dash='dot')))
                fig.update_layout(template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Tabel Hasil
                st.markdown("#### 📋 Tabel Rincian Angka Prediksi")
                df_res = pd.DataFrame({
                    'Tanggal': y.index,
                    'Data Aktual': y.values,
                    'Hasil Prediksi AI': preds.astype(int),
                    'Selisih (Error)': (preds - y.values).astype(int)
                }).sort_values(by='Tanggal', ascending=False)
                st.dataframe(df_res, use_container_width=True, height=300)

                # Download
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Tabel (CSV)", data=csv, file_name="forecast_result.csv", mime='text/csv')

        else:
            # --- MODE BREAKDOWN DENGAN PROGRESS BAR ---
            st.subheader(f"📊 Breakdown per Item ({selected_region})")
            unique_items = f_df['Item Type'].unique()[:5] # Limit Top 5 agar tidak lambat
            fig_br = go.Figure()
            
            progress_text = "Memproses rincian per item..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, item in enumerate(unique_items):
                # Update progress per item
                progress_val = int((i / len(unique_items)) * 100)
                my_bar.progress(progress_val, text=f"Menganalisis item: {item}...")
                
                item_df = f_df[f_df['Item Type'] == item].groupby(date_col)[selected_target].sum().reset_index()
                if len(item_df) > 20:
                    data_i = prepare_features(item_df, target_col=selected_target)
                    y_i = data_i[selected_target]
                    model_i, _ = train_optimized_xgb(data_i.drop(columns=[selected_target]), y_i)
                    preds_i = model_i.predict(data_i.drop(columns=[selected_target]))
                    fig_br.add_trace(go.Scatter(x=y_i.index, y=preds_i, name=f"{item}"))
            
            my_bar.progress(100, text="Analisis Breakdown Selesai!")
            time.sleep(1)
            my_bar.empty()
            
            st.plotly_chart(fig_br, use_container_width=True)
else:
    st.info("👋 Unggah file CSV untuk memulai.")
