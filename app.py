import streamlit as st
import pandas as pd
import plotly.express as px
import gdown  # Library untuk download dari Google Drive
import os # Untuk cek apakah file ada

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(
    page_title="Dashboard Kelelahan Jemaah",
    page_icon="🌙",
    layout="wide"
)

st.title("🕋 Dashboard Analisis Prediktif Kelelahan Jemaah Haji")
st.write("""
Aplikasi ini memvisualisasikan data bersih (4.3 juta baris) dari proyek *data science* untuk memprediksi kelelahan jemaah. 
Data ini telah diagregasi dan dibersihkan dari noise, siap untuk dianalisis.
""")

# --- Bagian 1: Pemuatan Data dari Google Drive ---

# ID file Google Drive (Ganti dengan ID file .parquet kamu)
# Link kamu: https://drive.google.com/file/d/THIS_IS_THE_ID/view?usp=sharing
FILE_ID = "https://drive.google.com/file/d/1zPbNjbejFsEcG7BeR5bTpCTOMTHujgvL/view?usp=sharing" 
FILE_PATH = "data_bersih_agregasi.parquet"

# Fungsi untuk mengunduh file jika belum ada
def download_data(file_id, output_path):
    if not os.path.exists(output_path):
        with st.spinner(f"Mengunduh data besar ({output_path})... Ini mungkin perlu 1-2 menit..."):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
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
        # Coba hapus file dan unduh ulang jika korup
        if os.path.exists(path):
            os.remove(path)
        st.write("Mencoba mengunduh ulang data...")
        download_data(FILE_ID, FILE_PATH)
        data = pd.read_parquet(path)
        return data

# Unduh dan muat data
download_data(FILE_ID, FILE_PATH)
df = load_data(FILE_PATH)

st.success(f"✅ Data bersih ({df.shape[0]} baris) berhasil dimuat.")

# --- Bagian 2: Tampilkan Metrik Utama ---
st.header("Metrik Utama Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Baris Data (Agregasi)", f"{df.shape[0]:,}")
col2.metric("Total Fitur (Sensor)", f"{df.shape[1] - 4}") # Kurangi id, createdAt, target
col3.metric("Jumlah Peserta (ID Unik)", f"{df['id'].nunique()}")


# --- Bagian 3: Visualisasi Interaktif ---
st.header("Eksplorasi Data Interaktif")

# Ambil sampel kecil untuk visualisasi agar lebih cepat
if st.checkbox("Gunakan sampel data (50.000 baris) agar dashboard lebih cepat"):
    df_sample = df.sample(50000, random_state=42)
else:
    df_sample = df.copy()

# 1. Distribusi Target
st.subheader("Distribusi Target (Lelah vs Tidak Lelah)")
target_dist = df_sample['y_binary'].value_counts().reset_index()
target_dist['y_binary'] = target_dist['y_binary'].map({0: '0 (Tidak Lelah)', 1: '1 (Lelah)'})
fig1 = px.pie(target_dist, names='y_binary', values='count', title="Proporsi Jemaah Lelah vs Tidak Lelah")
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
st.write("Ini adalah 100 baris pertama dari data bersih yang telah diagregasi.")
st.dataframe(df.head(100))

st.sidebar.header("Tentang Proyek")
st.sidebar.info("""
Proyek ini dibuat sebagai *final project* untuk menganalisis dan memprediksi kelelahan jemaah haji.

- **Data:** 5 juta baris data mentah dari 17 peserta, dibersihkan menjadi 4.3 juta baris.
- **Model:** LightGBM (LGBM) dipilih sebagai model terbaik (F1-Score 0.72) setelah mengalahkan Logistic Regression dan Random Forest.
- **Notebook:** [Link ke GitHub Notebook Anda]
""")
