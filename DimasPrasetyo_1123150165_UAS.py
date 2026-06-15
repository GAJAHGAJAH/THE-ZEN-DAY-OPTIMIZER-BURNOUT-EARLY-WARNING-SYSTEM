# =========================================================
# PROYEK UAS JST: THE ZEN-DAY OPTIMIZER
# Nama: Dimas Prasetyo
# NIM : 1123150165
# Kelas: [SESUAIKAN KELAS ANDA, MISAL: TI23SE1]
# Deskripsi: Sistem Peringatan Dini Burnout menggunakan Perceptron & Backpropagation
# UI/UX  : Direvitalisasi dengan tema "Zen Dark" - clean, modern, dashboard-style
# =========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI HALAMAN (harus dipanggil pertama kali)
# =========================================================
st.set_page_config(page_title="Zen-Day Optimizer", page_icon="🧠", layout="wide")

# =========================================================
# CUSTOM CSS - TEMA "ZEN DARK"
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0B0F19;
    --bg-secondary: #11182B;
    --card-bg: #161D2F;
    --card-border: rgba(255,255,255,0.07);
    --accent-teal: #5EEAD4;
    --accent-violet: #A78BFA;
    --text-primary: #E5E9F0;
    --text-secondary: #8B95AB;
    --success: #4ADE80;
    --warning: #FBBF24;
    --danger: #F87171;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 15% -10%, #1B2540 0%, #0B0F19 45%), #0B0F19;
}

h1, h2, h3, h4, h5 {
    font-family: 'Poppins', sans-serif !important;
}

/* ---------- HEADER ---------- */
.zen-header {
    position: relative;
    padding: 2.2rem 2.4rem;
    border-radius: 20px;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, rgba(94,234,212,0.10), rgba(167,139,250,0.10));
    border: 1px solid var(--card-border);
    overflow: hidden;
}
.zen-header::before {
    content: "";
    position: absolute;
    top: -70px; right: -70px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(94,234,212,0.35), transparent 70%);
    filter: blur(15px);
    pointer-events: none;
}
.zen-header::after {
    content: "";
    position: absolute;
    bottom: -90px; left: 10%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(167,139,250,0.25), transparent 70%);
    filter: blur(20px);
    pointer-events: none;
}
.zen-header h1 {
    font-weight: 700;
    font-size: 2.1rem;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #5EEAD4, #A78BFA);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
}
.zen-header p {
    color: var(--text-secondary);
    font-size: 0.98rem;
    margin: 0;
    max-width: 760px;
    position: relative;
    z-index: 1;
    line-height: 1.6;
}

/* ---------- GENERIC CARD ---------- */
.zen-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}
.zen-card-empty {
    text-align: center;
    padding: 3rem 1.5rem;
}
.zen-card-empty h3 { margin: 0 0 0.4rem 0; color: var(--text-primary); }
.zen-card-empty p { margin: 0; color: var(--text-secondary); }

/* ---------- INPUT SUMMARY METRIC ---------- */
.input-metric {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color 0.2s ease;
}
.input-metric:hover { border-color: rgba(94,234,212,0.35); }
.input-metric .icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
.input-metric .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    color: var(--accent-teal);
}
.input-metric .lbl {
    font-size: 0.8rem; color: var(--text-secondary);
    margin-top: 0.2rem;
}

/* ---------- STATUS / DASHBOARD CARDS ---------- */
.status-card {
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    border: 1px solid;
    height: 100%;
}
.status-safe    { background: rgba(74,222,128,0.06);  border-color: rgba(74,222,128,0.30); }
.status-warning { background: rgba(251,191,36,0.06);  border-color: rgba(251,191,36,0.30); }
.status-danger  { background: rgba(248,113,113,0.06); border-color: rgba(248,113,113,0.30); }

.status-card h3 { margin: 0.2rem 0 0.5rem 0; font-size: 1.08rem; }
.status-card p  { margin: 0; color: var(--text-secondary); font-size: 0.92rem; line-height: 1.55; }

.badge {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}
.badge-safe    { background: rgba(74,222,128,0.15);  color: var(--success); }
.badge-warning { background: rgba(251,191,36,0.15);  color: var(--warning); }
.badge-danger  { background: rgba(248,113,113,0.15); color: var(--danger); }

.score-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.1;
    margin-top: 0.6rem;
}

/* progress track for burnout score */
.score-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    margin: 0.7rem 0 0.9rem 0;
    overflow: hidden;
}
.score-fill { height: 100%; border-radius: 999px; }

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: var(--text-primary) !important;
}
.sidebar-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: var(--accent-teal);
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    border: 1px solid var(--card-border);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(94,234,212,0.16), rgba(167,139,250,0.16)) !important;
    border-color: rgba(94,234,212,0.35) !important;
    color: var(--text-primary) !important;
}

/* ---------- BUTTONS ---------- */
.stButton button {
    border-radius: 12px;
    background: linear-gradient(135deg, #5EEAD4, #A78BFA);
    color: #0B0F19;
    border: none;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    padding: 0.65rem 1.2rem;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(94,234,212,0.25);
}
.stButton button:active { transform: translateY(0px); }

/* ---------- MISC ---------- */
hr { border-color: var(--card-border) !important; }

.zen-footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--card-border);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BAGIAN 1: MODEL JST 1 - PERCEPTRON (Klasifikasi Sederhana)
# ---------------------------------------------------------
class ModelPerceptron:
    def __init__(self):
        # Bobot untuk 2 input: [Jam Tidur, Jam Layar]
        self.bobot = np.random.uniform(-0.5, 0.5, 2)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.lr = 0.1 

    def aktivasi(self, x):
        return 1 if x >= 0 else 0

    def prediksi(self, input_data):
        hasil = np.dot(input_data, self.bobot) + self.bias
        return self.aktivasi(hasil)

    def latih(self, data_latih, target, epochs=100, lr=0.1):
        self.lr = lr
        for _ in range(epochs):
            for i in range(len(data_latih)):
                x = data_latih[i]
                y = target[i]
                prediksi = self.prediksi(x)
                error = y - prediksi
                # Update bobot: w = w + (lr * error * x)
                self.bobot += self.lr * error * x
                self.bias += self.lr * error

# ---------------------------------------------------------
# BAGIAN 2: MODEL JST 2 - BACKPROPAGATION (Prediksi Skor Burnout)
# ---------------------------------------------------------
class ModelBackpropagation:
    def __init__(self):
        # 4 Input: [Tidur, Kafein, Belajar, Stres] -> 3 Hidden -> 1 Output
        self.w1 = np.random.uniform(-1, 1, (4, 3)) 
        self.b1 = np.random.uniform(-1, 1, (1, 3)) 
        self.w2 = np.random.uniform(-1, 1, (3, 1)) 
        self.b2 = np.random.uniform(-1, 1, (1, 1)) 
        self.sejarah_error = [] 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250))) # clip untuk mencegah overflow

    def sigmoid_turunan(self, x):
        return x * (1 - x)

    def latih(self, X, y, epochs=2000, lr=0.1):
        self.sejarah_error = []
        for epoch in range(epochs):
            # 1. FORWARD PROPAGATION
            hidden_input = np.dot(X, self.w1) + self.b1
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output, self.w2) + self.b2
            prediksi = self.sigmoid(output_input)
            
            # 2. HITUNG ERROR (MSE)
            error = y - prediksi
            mse = np.mean(np.square(error))
            self.sejarah_error.append(mse)
            
            # 3. BACKWARD PROPAGATION
            delta_output = error * self.sigmoid_turunan(prediksi)
            error_hidden = np.dot(delta_output, self.w2.T)
            delta_hidden = error_hidden * self.sigmoid_turunan(hidden_output)
            
            # 4. UPDATE BOBOT (w = w + lr * (input.T dot delta))
            self.w2 += np.dot(hidden_output.T, delta_output) * lr
            self.b2 += np.sum(delta_output, axis=0, keepdims=True) * lr
            self.w1 += np.dot(X.T, delta_hidden) * lr
            self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * lr

    def prediksi_skor(self, input_data):
        hidden_input = np.dot(input_data, self.w1) + self.b1
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.w2) + self.b2
        hasil = self.sigmoid(output_input)
        return hasil[0][0]

# ---------------------------------------------------------
# BAGIAN 3: APLIKASI STREAMLIT (Tampilan Antarmuka)
# ---------------------------------------------------------

# Inisialisasi session state untuk menyimpan hasil analisis antar-tab
if "hasil_analisis" not in st.session_state:
    st.session_state.hasil_analisis = None

# ---------- HEADER ----------
st.markdown("""
<div class="zen-header">
    <h1>🧠 The Zen-Day Optimizer</h1>
    <p>
        Sistem Peringatan Dini <b>Burnout</b> berbasis Jaringan Saraf Tiruan.
        Aplikasi ini menggabungkan dua model JST &mdash; <b>Perceptron</b> untuk pengecekan cepat
        dan <b>Backpropagation</b> untuk analisis mendalam &mdash; guna mendeteksi risiko
        kelelahan mental pada mahasiswa berdasarkan pola tidur, layar, kafein, belajar, dan stres.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="sidebar-eyebrow">Konfigurasi Model</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Parameter Backpropagation")
    lr_bp = st.slider("Learning Rate (Backprop)", 0.01, 1.0, 0.1, step=0.01)
    epoch_bp = st.slider("Jumlah Epoch", 500, 5000, 2000, step=500)
    st.caption("💡 Ubah nilai di atas untuk melihat pengaruhnya terhadap grafik error pada tab "
               "**Analisis Matematis & Grafik**.")

    st.divider()

    st.markdown('<div class="sidebar-eyebrow">Input Harian</div>', unsafe_allow_html=True)
    st.markdown("### 👤 Data Aktivitas Anda")
    jam_tidur = st.slider("🌙 Jam Tidur Malam Ini", 0.0, 12.0, 6.0, step=0.5)
    jam_layar = st.slider("📱 Jam Menatap Layar (HP/Laptop)", 0.0, 16.0, 8.0, step=0.5)
    kafein = st.slider("☕ Konsumsi Kafein (Cangkir)", 0.0, 5.0, 2.0, step=0.5)
    jam_belajar = st.slider("📖 Jam Belajar / Kerja", 0.0, 12.0, 4.0, step=0.5)
    tingkat_stres = st.slider("🔥 Tingkat Stres Subjektif (1-10)", 1, 10, 5)

    st.divider()

    analisis_clicked = st.button("🔍 Analisis Risiko Burnout", use_container_width=True)
    st.caption("Hasil akan muncul pada tab **📈 Hasil Analisis**.")

# ---------- TABS UTAMA ----------
tab1, tab2, tab3 = st.tabs(["📊 Input Data", "📈 Hasil Analisis", "🧮 Analisis Matematis & Grafik"])

# ===========================================================
# TAB 1: INPUT DATA
# ===========================================================
with tab1:
    st.markdown("#### Ringkasan Data Aktivitas Harian")
    st.caption("Data berikut diambil dari slider pada sidebar dan akan menjadi input untuk kedua model JST.")

    c1, c2, c3, c4, c5 = st.columns(5)
    ringkasan = [
        (c1, "🌙", f"{jam_tidur:.1f} jam", "Jam Tidur"),
        (c2, "📱", f"{jam_layar:.1f} jam", "Jam Layar"),
        (c3, "☕", f"{kafein:.1f} cup", "Kafein"),
        (c4, "📖", f"{jam_belajar:.1f} jam", "Jam Belajar"),
        (c5, "🔥", f"{tingkat_stres}/10", "Tingkat Stres"),
    ]
    for col, icon, val, lbl in ringkasan:
        with col:
            st.markdown(f"""
            <div class="input-metric">
                <div class="icon">{icon}</div>
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        <div class="zen-card">
            <h4>🛡️ Tentang Model 1 — Perceptron</h4>
            <p style="color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                Model klasifikasi sederhana yang menggunakan 2 input (Jam Tidur &amp; Jam Layar)
                untuk memberikan cek cepat: apakah pola harianmu tergolong <b>AMAN</b> atau
                <b>WASPADA</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="zen-card">
            <h4>🧠 Tentang Model 2 — Backpropagation</h4>
            <p style="color:var(--text-secondary); font-size:0.9rem; line-height:1.6;">
                Model jaringan saraf 4-3-1 (4 input &rarr; 3 hidden neuron &rarr; 1 output)
                yang menghitung <b>skor risiko burnout</b> secara lebih mendalam berdasarkan
                tidur, kafein, jam belajar, dan tingkat stres.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Sesuaikan slider pada sidebar, lalu klik **🔍 Analisis Risiko Burnout** untuk menjalankan kedua model JST.")

# ===========================================================
# PROSES ANALISIS (dijalankan saat tombol diklik)
# ===========================================================
if analisis_clicked:
    with st.spinner("Sedang memproses data dengan Algoritma JST..."):

        # Normalisasi Data (0-1)
        norm_tidur = jam_tidur / 12.0
        norm_layar = jam_layar / 16.0
        norm_kafein = kafein / 5.0
        norm_belajar = jam_belajar / 12.0
        norm_stres = tingkat_stres / 10.0

        # ==========================================
        # EKSEKUSI MODEL 1: PERCEPTRON
        # ==========================================
        data_latih_p = np.array([[0.8, 0.2], [0.3, 0.8], [0.9, 0.3], [0.4, 0.7]])
        target_p = np.array([1, 0, 1, 0])
        model_p = ModelPerceptron()
        model_p.latih(data_latih_p, target_p, epochs=50, lr=0.1)
        hasil_p = model_p.prediksi(np.array([norm_tidur, norm_layar]))

        # ==========================================
        # EKSEKUSI MODEL 2: BACKPROPAGATION
        # ==========================================
        X_latih = np.array([[0.8, 0.2, 0.3, 0.2], [0.3, 0.8, 0.9, 0.9], [0.7, 0.4, 0.5, 0.4], [0.4, 0.6, 0.8, 0.7]])
        y_latih = np.array([[0.0], [0.9], [0.2], [0.8]])

        model_bp = ModelBackpropagation()
        model_bp.latih(X_latih, y_latih, epochs=epoch_bp, lr=lr_bp) # Menggunakan parameter dari slider!

        input_bp = np.array([[norm_tidur, norm_kafein, norm_belajar, norm_stres]])
        skor_burnout = model_bp.prediksi_skor(input_bp) * 100

        # Simpan hasil ke session_state agar bisa dibaca oleh Tab 2 & Tab 3
        st.session_state.hasil_analisis = {
            "hasil_p": hasil_p,
            "skor_burnout": skor_burnout,
            "sejarah_error": model_bp.sejarah_error,
            "lr_bp": lr_bp,
            "epoch_bp": epoch_bp,
        }

    st.toast("✅ Analisis selesai! Cek tab 'Hasil Analisis' & 'Analisis Matematis'.")

# ===========================================================
# TAB 2: HASIL ANALISIS
# ===========================================================
with tab2:
    if st.session_state.hasil_analisis is None:
        st.markdown("""
        <div class="zen-card zen-card-empty">
            <h3>Belum Ada Hasil Analisis</h3>
            <p>Atur data aktivitasmu pada sidebar, lalu klik <b>🔍 Analisis Risiko Burnout</b> untuk
            melihat dashboard hasil di sini.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        hasil = st.session_state.hasil_analisis
        hasil_p = hasil["hasil_p"]
        skor_burnout = hasil["skor_burnout"]

        col1, col2 = st.columns(2)

        # ---------- MODEL 1: PERCEPTRON ----------
        with col1:
            st.markdown("#### 🛡️ Model 1: Perceptron (Cek Cepat)")
            if hasil_p == 1:
                st.markdown("""
                <div class="status-card status-safe">
                    <span class="badge badge-safe">Status: Aman</span>
                    <h3>✅ Pola Aktivitas Normal</h3>
                    <p>Kombinasi jam tidur dan screen time Anda masih dalam batas wajar.
                    Pertahankan rutinitas ini!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card status-warning">
                    <span class="badge badge-warning">Status: Waspada</span>
                    <h3>⚠️ Perlu Perhatian</h3>
                    <p>Kombinasi kurang tidur dan screen time tinggi terdeteksi.
                    Coba kurangi waktu di depan layar sebelum tidur.</p>
                </div>
                """, unsafe_allow_html=True)

        # ---------- MODEL 2: BACKPROPAGATION ----------
        with col2:
            st.markdown("#### 🧠 Model 2: Backpropagation (Analisis Mendalam)")

            if skor_burnout < 40:
                kelas, badge, judul, pesan, warna = (
                    "status-safe", "badge-safe", "Risiko Rendah",
                    "Pertahankan rutinitas sehat Anda!", "var(--success)"
                )
            elif skor_burnout < 70:
                kelas, badge, judul, pesan, warna = (
                    "status-warning", "badge-warning", "Risiko Sedang",
                    "Pertimbangkan untuk istirahat sejenak dari aktivitas Anda.", "var(--warning)"
                )
            else:
                kelas, badge, judul, pesan, warna = (
                    "status-danger", "badge-danger", "🚨 Risiko Tinggi",
                    "Tanda-tanda burnout terdeteksi. Sangat disarankan untuk berhenti "
                    "bekerja sejenak dan beristirahat.", "var(--danger)"
                )

            lebar = min(max(skor_burnout, 0), 100)
            st.markdown(f"""
            <div class="status-card {kelas}">
                <span class="badge {badge}">{judul}</span>
                <div class="score-display" style="color:{warna};">{skor_burnout:.1f}%</div>
                <div class="score-track">
                    <div class="score-fill" style="width:{lebar}%; background:{warna};"></div>
                </div>
                <p>{pesan}</p>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================
# TAB 3: ANALISIS MATEMATIS & GRAFIK
# ===========================================================
with tab3:
    if st.session_state.hasil_analisis is None:
        st.markdown("""
        <div class="zen-card zen-card-empty">
            <h3>Grafik Belum Tersedia</h3>
            <p>Jalankan analisis terlebih dahulu dari sidebar untuk melihat grafik penurunan
            error (MSE) dan contoh perhitungan matematis di sini.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        hasil = st.session_state.hasil_analisis

        # ==========================================
        # GRAFIK MSE & ANALISIS PARAMETER (Syarat Bab IV)
        # ==========================================
        st.markdown("#### 📈 Grafik Penurunan Mean Squared Error (MSE)")
        st.caption(
            f"Grafik ini menunjukkan penurunan error dengan "
            f"Learning Rate = **{hasil['lr_bp']}** dan Epoch = **{hasil['epoch_bp']}**."
        )

        with st.container(border=True):
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor('#161D2F')
            ax.set_facecolor('#161D2F')

            ax.plot(hasil["sejarah_error"], color='#5EEAD4', linewidth=2.4)

            ax.set_title("Grafik Penurunan Mean Squared Error (MSE) Selama Training",
                          color='#E5E9F0', fontsize=12, pad=12)
            ax.set_xlabel("Epoch (Iterasi)", color='#8B95AB')
            ax.set_ylabel("Nilai Error (MSE)", color='#8B95AB')
            ax.tick_params(colors='#8B95AB')
            ax.grid(True, linestyle='--', alpha=0.15, color='#8B95AB')
            for spine in ax.spines.values():
                spine.set_color('#2A3349')

            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("")

        # ==========================================
        # CONTOH PERHITUNGAN MATEMATIS (Syarat Bab III)
        # ==========================================
        with st.expander("🧮 Klik di sini untuk melihat Contoh Perhitungan Matematis (1 Iterasi)"):
            st.markdown("""
            **Contoh Perhitungan Manual 1 Iterasi (Forward & Backward Propagation)**  
            *Misalkan kita memiliki 1 data input sederhana yang sudah dinormalisasi:*
            - Input (X) = `[0.5, 0.5, 0.5, 0.5]` (Tidur, Kafein, Belajar, Stres)
            - Target (Y) = `[0.6]`
            - Bobot Acak Awal (W1) = `0.2` (disederhanakan untuk contoh)
            - Learning Rate (lr) = `0.1`

            **1. Forward Propagation (Maju):**
            - `Hidden_Input` = (X • W1) + Bias = (0.5 * 0.2) + 0 = **0.1**
            - `Hidden_Output` = Sigmoid(0.1) = 1 / (1 + e^-0.1) ≈ **0.525**
            - `Output_Prediksi` = Sigmoid(Hidden_Output • W2 + Bias) ≈ **0.55**

            **2. Menghitung Error:**
            - `Error` = Target - Prediksi = 0.6 - 0.55 = **0.05**
            - `MSE` = (0.05)² = **0.0025**

            **3. Backward Propagation (Mundur / Update Bobot):**
            - `Delta_Output` = Error * Turunan_Sigmoid(Prediksi) = 0.05 * (0.55 * (1 - 0.55)) ≈ **0.0123**
            - `Update Bobot (W2)` = W2_lama + (lr * Hidden_Output * Delta_Output)  
              = 0.2 + (0.1 * 0.525 * 0.0123) ≈ **0.2006** *(Bobot diperbarui agar error berikutnya lebih kecil!)*
            
            *Proses ini diulang sebanyak 'Epoch' kali hingga nilai MSE mendekati 0.*
            """)

# ---------- FOOTER ----------
st.markdown("""
<div class="zen-footer">
    🧠 The Zen-Day Optimizer &mdash; Proyek UAS Jaringan Saraf Tiruan &middot;
    Dimas Prasetyo &middot; NIM 1123150165
</div>
""", unsafe_allow_html=True)
