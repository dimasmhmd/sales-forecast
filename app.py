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
st.markdown("Analisis prediktif multivariat dengan **Loading Bar**, **Tabel Rincian**, & **Breakdown Analysis**.")

# --- 2. SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📁 Manajemen Data")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Deteksi Kolom Tanggal
    date_cols = [c for c in df.columns if 'Date' in c or 'tanggal' in c.lower()]
    if not date_cols:
        st.error("❌ Tidak ditemukan kolom tanggal. Pastikan kolom memiliki kata 'Date'.")
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

    # Preview Data Terfilter
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
            
            # Agregasi Harian (Penting agar tidak ada duplikat tanggal)
            f_df_agg = f_df.groupby(date_col)[selected_target].sum().reset_index()
            
            if len(f_df_agg) < 20:
                st.warning("⚠️ Data terlalu sedikit untuk dianalisis (Min. 20 hari data).")
            else:
                # --- LOADING BAR START ---
                progress_text = "Memulai optimasi AI..."
                my_bar = st.progress(0, text=progress_text)
                
                my_bar.progress(30, text="Melakukan Feature Engineering...")
                data_final = prepare_features(f_df_agg, target_col=selected_target)
                X, y = data_final.drop(columns=[selected_target]), data_final[selected_target]
                
                my_bar.progress(60, text="Mencari Hyperparameter terbaik (XGBoost)...")
                model, _ = train_optimized_xgb(X, y)
                
                my_bar.progress(90, text="Menghasilkan angka prediksi...")
                preds = model.predict(X)
                
                time.sleep(0.5)
                my_bar.empty() 
                # --- LOADING BAR END ---

                # --- TAMPILAN DASHBOARD HASIL ---
                st.divider()
                st.subheader(f"📊 Hasil Forecast: {selected_item} ({selected_region})")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediksi Berikutnya", f"{int(preds[-1]):,}")
                m2.metric("Total Sampel Harian", len(y))
                m3.metric("Safety Stock (120%)", f"{int(preds[-1] * 1.2):,}")

                # 1. Grafik Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index, y=y, name="Data Aktual", line=dict(color='#1f77b4', width=2)))
                fig.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi AI", line=dict(color='#ff7f0e', dash='dot')))
                fig.update_layout(template="plotly_white", hovermode="x unified", title=f"Tren {selected_target} Over Time")
                st.plotly_chart(fig, use_container_width=True)

                # --- 2. TABEL HASIL (PASTIKAN BAGIAN INI ADA) ---
                st.markdown("### 📋 Rincian Data Prediksi")
                df_res = pd.DataFrame({
                    'Tanggal': y.index,
                    'Aktual': y.values.astype(int),
                    'Prediksi': preds.astype(int),
                    'Selisih': (preds - y.values).astype(int)
                }).sort_values(by='Tanggal', ascending=False) # Data terbaru di atas

                st.dataframe(df_res, use_container_width=True, height=350)

                # 3. Tombol Download
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Tabel (.csv)",
                    data=csv,
                    file_name=f"forecast_{selected_item}.csv",
                    mime='text/csv'
                )

        else:
            # --- MODE BREAKDOWN PER ITEM ---
            st.subheader(f"📊 Breakdown Prediksi per Item ({selected_region})")
            unique_items = f_df['Item Type'].unique()[:5]
            fig_br = go.Figure()
            
            my_bar = st.progress(0, text="Memproses Breakdown...")
            for i, item in enumerate(unique_items):
                item_df = f_df[f_df['Item Type'] == item].groupby(date_col)[selected_target].sum().reset_index()
                if len(item_df) > 20:
                    data_i = prepare_features(item_df, target_col=selected_target)
                    y_i = data_i[selected_target]
                    model_i, _ = train_optimized_xgb(data_i.drop(columns=[selected_target]), y_i)
                    preds_i = model_i.predict(data_i.drop(columns=[selected_target]))
                    fig_br.add_trace(go.Scatter(x=y_i.index, y=preds_i, name=f"{item}"))
                my_bar.progress((i + 1) / len(unique_items))
            
            time.sleep(0.5)
            my_bar.empty()
            st.plotly_chart(fig_br, use_container_width=True)
else:
    st.info("👋 Unggah file CSV untuk memulai analisis.")
