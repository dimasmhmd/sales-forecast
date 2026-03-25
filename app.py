import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.processor import prepare_features
from src.trainer import train_optimized_xgb

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sales Analytics Pro", 
    layout="wide", 
    page_icon="📈"
)

st.title("🌎 Global Sales Forecasting Dashboard")
st.markdown("Analisis prediktif multivariat menggunakan **XGBoost dengan Hyperparameter Tuning**.")

# --- 2. SIDEBAR: UPLOAD & FILTER ---
st.sidebar.header("📁 Manajemen Data")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset Penjualan (CSV)", type="csv")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Deteksi Kolom Tanggal Otomatis (Cari kolom yang mengandung kata 'Date')
    date_cols = [c for c in df.columns if 'Date' in c]
    if not date_cols:
        st.error("❌ Tidak ditemukan kolom tanggal. Pastikan ada kolom dengan kata 'Date'.")
        st.stop()
    
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # --- FILTER DINAMIS ---
    st.sidebar.subheader("🎯 Filter Analisis")
    
    # Filter Region (Wilayah)
    selected_region = "All"
    if 'Region' in df.columns:
        regions = ["All"] + sorted(df['Region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Wilayah (Region)", regions)

    # Filter Item Type (Jenis Barang)
    selected_item = "All"
    if 'Item Type' in df.columns:
        items = ["All"] + sorted(df['Item Type'].unique().tolist())
        selected_item = st.sidebar.selectbox("Jenis Barang", items)

    # Terapkan Filter ke DataFrame
    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_item != "All":
        filtered_df = filtered_df[filtered_df['Item Type'] == selected_item]

    # Pilih Target Prediksi (Hanya kolom angka)
    target_options = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    default_target = 'Units Sold' if 'Units Sold' in target_options else target_options[0]
    selected_target = st.sidebar.selectbox("Target Prediksi", target_options, index=target_options.index(default_target))

    # --- 3. DISPLAY DATA MENTAH (EXPANDER) ---
    with st.expander("👁️ Lihat Data Terfilter (Raw Data)"):
        st.info(f"Menampilkan {len(filtered_df)} baris data.")
        st.dataframe(filtered_df, use_container_width=True, height=350)

    # --- 4. PROSES FORECASTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Jalankan Forecasting"):
        if len(filtered_df) < 25:
            st.warning("⚠️ Data terlalu sedikit untuk membuat prediksi (Min. 25 baris). Coba kurangi filter.")
        else:
            with st.spinner('Mengoptimalkan model AI...'):
                
                # Feature Engineering
                data_final = prepare_features(filtered_df, target_col=selected_target)
                
                # Split Feature & Target
                X = data_final.drop(columns=[selected_target])
                y = data_final[selected_target]
                
                # Training & Tuning
                model, best_params = train_optimized_xgb(X, y)
                preds = model.predict(X)

                # --- 5. DASHBOARD HASIL ---
                st.divider()
                st.subheader(f"📊 Hasil Analisis: {selected_item} ({selected_region})")
                
                # Metriks Utama
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediksi Berikutnya", f"{preds[-1]:,.0f}")
                m2.metric("Total Data Terolah", len(y))
                m3.metric("Rekomendasi Stok", f"{(preds[-1] * 1.2):,.0f}", delta="Safety Stock 20%")

                # Grafik Tren Plotly
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=y.index, y=y, name="Data Aktual", line=dict(color='#1f77b4', width=2.5)))
                fig_trend.add_trace(go.Scatter(x=y.index, y=preds, name="Prediksi AI", line=dict(color='#ff7f0e', dash='dot')))
                fig_trend.update_layout(
                    title=f"Tren {selected_target} Over Time",
                    xaxis_title="Waktu",
                    yaxis_title="Nilai",
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # --- 6. FEATURE IMPORTANCE ---
                st.subheader("🧠 Faktor Penentu Penjualan")
                importance = pd.DataFrame({
                    'Fitur': X.columns, 
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True).tail(10)
                
                fig_imp = px.bar(
                    importance, x='Importance', y='Fitur', orientation='h',
                    title="10 Variabel Paling Berpengaruh",
                    color='Importance', color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                # --- 7. DOWNLOAD HASIL ---
                st.divider()
                st.subheader("📥 Unduh Hasil Prediksi")
                
                df_res = pd.DataFrame({
                    'Tanggal': y.index,
                    'Aktual': y.values,
                    'Prediksi': preds,
                    'Selisih': preds - y.values
                })

                @st.cache_data
                def convert_df(df_input):
                    return df_input.to_csv(index=False).encode('utf-8')

                csv = convert_df(df_res)
                st.download_button(
                    label="💾 Download CSV Hasil Prediksi",
                    data=csv,
                    file_name=f'forecast_{selected_item}_{selected_region}.csv',
                    mime='text/csv',
                )
                
                st.success(f"Analisis Selesai! Model menggunakan parameter: {best_params}")

else:
    st.info("👋 Silakan unggah file CSV Global Sales Anda di sidebar untuk memulai.")
