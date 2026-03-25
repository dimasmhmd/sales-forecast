import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sales Analytics Pro", layout="wide", page_icon="📈")

st.title("🌎 Global Sales Forecasting Dashboard")
st.markdown("Analisis prediktif multivariat dengan fitur **Breakdown Analysis** & **Data Export**.")

# --- 2. SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📁 Manajemen Data")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Deteksi Kolom Tanggal Otomatis
    date_cols = [c for c in df.columns if 'Date' in c or 'tanggal' in c.lower()]
    if not date_cols:
        st.error("❌ Tidak ditemukan kolom tanggal. Mohon cek file CSV Anda.")
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

    # Opsi Tampilan khusus jika pilih All
    view_mode = "Total Agregat"
    if selected_item == "All":
        view_mode = st.sidebar.radio("Mode Tampilan Forecast:", ["Total Agregat", "Breakdown per Item"])

    # Pilih Target Prediksi (Kolom Angka)
    target_options = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    selected_target = st.sidebar.selectbox("Pilih Target Prediksi", target_options, index=0)

    # Filter Data Awal
    f_df = df.copy()
    if selected_region != "All":
        f_df = f_df[f_df['Region'] == selected_region]
    
    # --- 3. FITUR PREVIEW DATA (SCROLLABLE) ---
    with st.expander("👁️ Preview Data Terfilter"):
        # Jika user pilih item spesifik, filter preview-nya juga
        preview_df = f_df.copy()
        if selected_item != "All":
            preview_df = preview_df[preview_df['Item Type'] == selected_item]
        
        st.write(f"Menampilkan {len(preview_df)} baris data.")
        st.dataframe(preview_df, use_container_width=True, height=350)

    # --- 4. PROSES FORECASTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Jalankan Forecasting"):
        
        if view_mode == "Total Agregat":
            # --- MODE 1: TOTAL AGREGAT ---
            # Jika user pilih item spesifik, filter dulu baru agregasi harian
            if selected_item != "All":
                f_df = f_df[f_df['Item Type'] == selected_item]
            
            # Agregasi: Menjumlahkan sales per tanggal (agar tidak ada duplikat tanggal)
            f_df_agg = f_df.groupby(date_col)[selected_target].sum().reset_index()
            
            if len(f_df_agg) < 20:
                st.warning("⚠️ Data terlalu sedikit untuk agregasi harian. Coba kurangi filter.")
            else:
                with st.spinner(f'Menganalisis Total {selected_target}...'):
                    data_final = prepare_features(f_df_agg, target_col=selected_target)
                    X, y = data_final.drop(columns=[selected_target]), data_final[selected_target]
                    model, best_params = train_optimized_xgb(X, y)
                    preds = model.predict(X)

                    # Hasil Dashboard
                    st.divider()
                    st.subheader(f"📊 Forecast Total: {selected_item} ({selected_region})")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prediksi Berikutnya", f"{preds[-1]:,.0f}")
                    c2.metric("Total Sampel Harian", len(y))
                    c3.metric("Rekomendasi Stok", f"{(preds[-1] * 1.2):,.0f}")

                    # Grafik
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y.index, y=y, name="Aktual", line=dict(color='#1f77b4')))
                    fig.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi", line=dict(color='#ff7f0e', dash='dot')))
                    fig.update_layout(template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                    # --- FITUR DOWNLOAD ---
                    df_res = pd.DataFrame({'Tanggal': y.index, 'Aktual': y.values, 'Prediksi': preds})
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Hasil Forecast (CSV)", data=csv, file_name="forecast_total.csv", mime='text/csv')

        else:
            # --- MODE 2: BREAKDOWN PER ITEM ---
            st.subheader(f"📊 Breakdown Forecast per Item ({selected_region})")
            fig_br = go.Figure()
            
            # Ambil Top 5 Item berdasarkan volume tersering untuk menghindari grafik penuh
            unique_items = f_df['Item Type'].unique()[:5]
            
            progress = st.progress(0)
            for i, item in enumerate(unique_items):
                item_df = f_df[f_df['Item Type'] == item].groupby(date_col)[selected_target].sum().reset_index()
                
                if len(item_df) > 20:
                    data_i = prepare_features(item_df, target_col=selected_target)
                    X_i, y_i = data_i.drop(columns=[selected_target]), data_i[selected_target]
                    model_i, _ = train_optimized_xgb(X_i, y_i)
                    preds_i = model_i.predict(X_i)
                    
                    fig_br.add_trace(go.Scatter(x=y_i.index, y=preds_i, name=f"Prediksi {item}"))
                
                progress.progress((i + 1) / len(unique_items))
            
            fig_br.update_layout(title="Perbandingan Prediksi Antar Item", template="plotly_white")
            st.plotly_chart(fig_br, use_container_width=True)
            st.success("Analisis Breakdown Selesai!")

else:
    st.info("👋 Silakan unggah file CSV Global Sales Anda di sidebar untuk memulai.")
