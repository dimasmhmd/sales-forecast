import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Analytics Pro", layout="wide", page_icon="📈")

st.title("🌎 Global Sales Forecasting Dashboard")
st.markdown("Analisis prediktif multivariat dengan fitur **Breakdown Analysis**.")

# --- 2. SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📁 Manajemen Data")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    date_cols = [c for c in df.columns if 'Date' in c]
    if not date_cols:
        st.error("❌ Tidak ditemukan kolom tanggal.")
        st.stop()
    
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # --- FILTER SIDEBAR ---
    st.sidebar.subheader("🎯 Filter Analisis")
    
    selected_region = "All"
    if 'Region' in df.columns:
        regions = ["All"] + sorted(df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Wilayah (Region)", regions)

    selected_item = "All"
    if 'Item Type' in df.columns:
        items = ["All"] + sorted(df['Item Type'].unique().tolist())
        selected_item = st.sidebar.selectbox("Jenis Barang", items)

    # --- FITUR BARU: OPSI TAMPILAN JIKA PILIH ALL ---
    view_mode = "Total Agregat"
    if selected_item == "All":
        view_mode = st.sidebar.radio("Tampilan Forecast:", ["Total Agregat", "Breakdown per Item"])

    # Pilih Target
    target_options = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    selected_target = st.sidebar.selectbox("Target Prediksi", target_options, index=0)

    # Proses Filter Data
    f_df = df.copy()
    if selected_region != "All":
        f_df = f_df[f_df['Region'] == selected_region]
    if selected_item != "All" and view_mode == "Total Agregat":
        f_df = f_df[f_df['Item Type'] == selected_item]

    # --- 3. PROSES FORECASTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Jalankan Forecasting"):
        
        if view_mode == "Total Agregat":
            # LOGIKA 1: PREDIKSI TOTAL (Sama seperti sebelumnya)
            # Agregasi data berdasarkan tanggal jika ada duplikasi (misal beda negara di tanggal sama)
            f_df_agg = f_df.groupby(date_col)[selected_target].sum().reset_index()
            
            with st.spinner('Mengoptimalkan model untuk data agregat...'):
                data_final = prepare_features(f_df_agg, target_col=selected_target)
                X, y = data_final.drop(columns=[selected_target]), data_final[selected_target]
                model, _ = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # Visualisasi
                st.subheader(f"📊 Forecast Total {selected_target}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y.index, y=y, name="Aktual", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi", line=dict(color='#ff7f0e', dash='dot')))
                st.plotly_chart(fig, use_container_width=True)

        else:
            # LOGIKA 2: BREAKDOWN PER ITEM
            st.subheader(f"📊 Perbandingan Forecast per Jenis Barang ({selected_region})")
            fig_breakdown = go.Figure()
            
            # Ambil 5 item teratas saja agar grafik tidak penuh (opsional)
            top_items = f_df['Item Type'].unique()[:5] 
            
            progress_bar = st.progress(0)
            for i, item in enumerate(top_items):
                item_df = f_df[f_df['Item Type'] == item].groupby(date_col)[selected_target].sum().reset_index()
                
                if len(item_df) > 20:
                    data_f = prepare_features(item_df, target_col=selected_target)
                    X_i, y_i = data_f.drop(columns=[selected_target]), data_f[selected_target]
                    model_i, _ = train_optimized_xgb(X_i, y_i)
                    preds_i = model_i.predict(X_i)
                    
                    # Tambahkan ke grafik yang sama
                    fig_breakdown.add_trace(go.Scatter(x=y_i.index, y=preds_i, name=f"Prediksi {item}"))
                
                progress_bar.progress((i + 1) / len(top_items))
            
            fig_breakdown.update_layout(title="Breakdown Prediksi per Kategori Produk", template="plotly_white")
            st.plotly_chart(fig_breakdown, use_container_width=True)
            st.success("Analisis Breakdown Selesai!")

else:
    st.info("👋 Silakan unggah file CSV Global Sales Anda di sidebar.")
