import streamlit as st
import pandas as pd
import plotly.express as px
import wget  # <-- Menggunakan wget
import os
import joblib # <-- Tambahkan ini untuk memuat model .pkl
import lightgbm # <-- Tambahkan ini agar .pkl bisa dibaca
import sklearn # <-- Tambahkan ini agar .pkl bisa dibaca

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(
    page_title="Dashboard Kelelahan Jemaah",
    page_icon="🌙",
    layout="wide"
)

st.title("🕋 Dashboard Analisis Prediktif Kelelahan Jemaah Haji")
st.write("""
Dashboard ini memvisualisasikan **sampel 50.000 baris** dari dataset bersih proyek. 
Data asli (4.3 juta baris) telah diagregasi dan dibersihkan dari noise.
""")

# --- Bagian 1: Pemuatan Data dari Google Drive ---

# ID file untuk file SAMPEL 50k baris
FILE_ID = "1prQQkSUDcYltzPCtX5wcJmr4JbR9WPXS" 
FILE_PATH = "data_bersih_SAMPEL_50k.parquet"

# Fungsi untuk mengunduh file jika belum ada
def download_data_new(file_id, output_path):
    if not os.path.exists(output_path):
        with st.spinner(f"Mengunduh data sampel ({output_path})... Ini hanya sebentar..."):
            try:
                print(f"\nMenggunakan wget untuk mengunduh...")
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
                wget.download(url, out=output_path)
                print(f"\nDownload selesai.")
            except Exception as e:
                st.error(f"Download gagal: {e}")
                st.error("Pastikan file memiliki izin 'Siapa saja yang memiliki link'.")
                raise e
    else:
        print("Data sudah ada, tidak perlu mengunduh.")

# Fungsi untuk memuat data (dengan cache agar cepat)
@st.cache_data
def load_data(path):
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        if os.path.exists(path):
            os.remove(path)
        st.write("Mencoba mengunduh ulang data...")
        # MEMANGGIL FUNGSI YANG BENAR:
        download_data_new(FILE_ID, FILE_PATH) 
        data = pd.read_parquet(path)
        return data

# --- Eksekusi Pemuatan Data ---
# (Ini adalah panggilan utama)
download_data_new(FILE_ID, FILE_PATH) # <-- MEMANGGIL FUNGSI YANG BENAR
df = load_data(FILE_PATH)
# --- Selesai Pemuatan Data ---


st.success(f"✅ Data sampel ({df.shape[0]} baris) berhasil dimuat.")

# --- Bagian 2: Tampilkan Metrik Utama ---
st.header("Metrik Utama Dataset (dari Sampel 50k)")
col1, col2, col3 = st.columns(3)
col1.metric("Total Baris Data (Sampel)", f"{df.shape[0]:,}")
col2.metric("Total Fitur (Sensor)", f"{df.shape[1] - 4}") # Kurangi id, createdAt, target
col3.metric("Jumlah Peserta (ID Unik)", f"{df['id'].nunique()}")


# --- Bagian 3: Visualisasi Interaktif ---
st.header("Eksplorasi Data Interaktif")

# Kita sudah pakai sampel, jadi df_sample = df
df_sample = df.copy()

# 1. Distribusi Target
st.subheader("Distribusi Target (Lelah vs Tidak Lelah)")
target_dist = df_sample['y_binary'].value_counts().reset_index()
target_dist['y_binary'] = target_dist['y_binary'].map({0: '0 (Tidak Lelah)', 1: '1 (Lelah)'})
fig1 = px.pie(target_dist, names='y_binary', values='count', title="Proporsi Jemaah Lelah vs Tidak Lelah (di Sampel)")
st.plotly_chart(fig1, use_container_width=True)

# 2. Scatter Plot (Hubungan Antar Sensor)
st.subheader("Hubungan Antar Sensor")
col1_scatter, col2_scatter = st.columns([1, 3])

# Pilih fitur untuk di-plot
x_axis = col1_scatter.selectbox("Pilih Sumbu X", ['heartRate', 'skinTemperature', 'gsr_x', 'x', 'y', 'z'], index=0)
y_axis = col1_scatter.selectbox("Pilih Sumbu Y", ['heartRate', 'skinTemperature', 'gsr_x', 'x', 'y', 'z'], index=1)

fig2 = px.scatter(
    df_sample,
    x=x_axis,
    y=y_axis,
    color=df_sample['y_binary'].map({0: 'Tidak Lelah', 1: 'Lelah'}), # Warnai berdasarkan target
    title=f"Scatter Plot: {x_axis} vs {y_axis}",
    opacity=0.5 # Buat transparan karena data padat
)
col2_scatter.plotly_chart(fig2, use_container_width=True)

# 3. Distribusi Fitur
st.subheader("Distribusi Fitur Sensor")
feature = st.selectbox("Pilih Fitur untuk melihat distribusinya", ['heartRate', 'skinTemperature', 'gsr_x', 'x', 'y', 'z'])
fig3 = px.histogram(df_sample, x=feature, color=df_sample['y_binary'].map({0: 'Tidak Lelah', 1: 'Lelah'}),
                    title=f"Distribusi {feature} (Dipisah berdasarkan Target)",
                    marginal="box") # Tambahkan boxplot
st.plotly_chart(fig3, use_container_width=True)


# --- Bagian 4: Tampilkan Sampel Data ---
st.header("Sampel Data Bersih")
st.write("Ini adalah 100 baris pertama dari data sampel 50k.")
st.dataframe(df.head(100))

# --- Sidebar ---
st.sidebar.header("Tentang Proyek")
st.sidebar.info("""
Proyek ini dibuat sebagai *final project* untuk menganalisis dan memprediksi kelelahan jemaah haji.

- **Data:** Sampel 50k baris dari 4.3 juta data bersih.
- **Model:** LightGBM (LGBM) dipilih sebagai model terbaik (F1-Score 0.72) setelah mengalahkan Logistic Regression dan Random Forest.
- **Notebook:** [GANTI DENGAN LINK KE NOTEBOOK GITHUB ANDA]
""")

# Coba muat model .pkl untuk memastikan library ada
try:
    joblib.load('model_pemenang_LGBM.pkl')
    st.sidebar.success("Model .pkl berhasil dimuat.")
except FileNotFoundError:
    st.sidebar.warning("File model 'model_pemenang_LGBM.pkl' tidak ditemukan di repositori.")
except Exception as e:
    st.sidebar.error(f"Error memuat model .pkl: {e}")
