import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
st.set_page_config(page_title="SPK Paris | SAW", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(160deg, #0f0c29, #302b63, #24243e); }
    section[data-testid="stSidebar"] * { color: #e0e0ff !important; }
    .hero { background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); border-radius: 14px;
            padding: 1.8rem 2.2rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.07); }
    .hero h1 { color: white; font-size: 1.7rem; margin: 0; font-weight: 800; }
    .hero p  { color: rgba(255,255,255,0.55); font-size: 0.9rem; margin: 0.3rem 0 0; }
    .shead   { font-size: 1rem; font-weight: 700; color: #1a1a2e; border-left: 4px solid #667eea;
               padding-left: 0.7rem; margin: 1.4rem 0 0.8rem; }
    .mcard   { background: linear-gradient(135deg, #1e1e3f, #2a2a5a); border: 1px solid rgba(102,126,234,0.25);
               border-radius: 12px; padding: 1rem; text-align: center; }
    .mcard .v { font-size: 1.6rem; font-weight: 800; color: #a5b4fc; }
    .mcard .l { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.2rem; }
    .pill    { display:inline-block; background: linear-gradient(135deg,#667eea,#764ba2); color:white;
               border-radius:50px; padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:700; }
    .pcard   { border-radius: 12px; padding: 1.2rem 1.5rem; color: white; }
    .pcard h3 { margin: 0 0 0.4rem; font-size: 1rem; }
    .pcard p  { margin: 0.15rem 0; font-size: 0.88rem; opacity: 0.9; }
    div.stButton > button { background: linear-gradient(135deg,#667eea,#764ba2) !important;
        color:white !important; border:none !important; font-weight:700 !important;
        padding:0.7rem 1.5rem !important; border-radius:10px !important; width:100% !important; }
</style>""", unsafe_allow_html=True)
 
# ── Konstanta ──────────────────────────────────────────────────────────────────
CRITERIA        = ["squareMeters","numberOfRooms","cityPartRange","numPrevOwners","hasGuestRoom","price"]
CRITERIA_LABELS = ["Luas (m²)","Jml. Kamar","Zona Kota","Pemilik Sblm","Kamar Tamu","Harga (€)"]
CRITERIA_TYPES  = ["Benefit","Benefit","Benefit","Cost","Benefit","Cost"]
DEFAULT_WEIGHTS = [9, 8, 6, 5, 5, 7]
 
FASIL = [
    {"col":"hasPool",          "label":"Kolam Renang",     "icon":"🏊"},
    {"col":"hasYard",          "label":"Halaman",           "icon":"🌿"},
    {"col":"hasStorageRoom",   "label":"Ruang Penyimpanan", "icon":"📦"},
    {"col":"hasStormProtector","label":"Pelindung Badai",   "icon":"🛡️"},
]
PALETTE = ["#667eea","#764ba2","#f64f59","#c471ed","#12c2e9","#f7971e"]
 
# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("ParisHousing_modified.csv")
    except FileNotFoundError:
        st.error("⚠️ File ParisHousing_modified.csv tidak ditemukan!")
        st.stop()
 
# ── SAW Core ───────────────────────────────────────────────────────────────────
def run_saw(df, weights):
    w = np.array(weights, dtype=float); w /= w.sum()
    norm = pd.DataFrame(index=df.index)
    for col, t in zip(CRITERIA, CRITERIA_TYPES):
        v = df[col].astype(float)
        norm[col] = v / v.max() if t == "Benefit" else v.min() / v.replace(0, np.nan).fillna(v.min())
    scores = (norm.values * w).sum(axis=1)
    result = df.copy().reset_index(drop=True)
    result["Skor SAW"] = np.round(scores, 6)
    result["Peringkat"] = result["Skor SAW"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("Skor SAW", ascending=False).reset_index(drop=True), norm, w
 
def apply_filter(df, prefs):
    for col, val in prefs.items():
        if val is not None and col in df.columns:
            df = df[df[col] == val]
    return df
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("<div style='text-align:center;padding:1rem 0 0.5rem'>"
        "<div style='font-size:2rem'>🏙️</div>"
        "<div style='font-size:1rem;font-weight:800'>SPK PARIS</div>"
        "<div style='font-size:0.72rem;opacity:0.55'>Simple Additive Weighting</div></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("📋 Navigasi", ["🏠  Beranda & Dataset","⚙️  Hitung SPK (SAW)","📊  Visualisasi Data","👥  Profil Kelompok"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.72rem;opacity:0.45;text-align:center'>SCPK 2025/2026 · SAW · Kaggle</div>", unsafe_allow_html=True)
    return menu
 
# ── Halaman 1: Beranda ─────────────────────────────────────────────────────────
def page_beranda(df):
    st.markdown("<div class='hero'><h1>🏙️ SPK Pemilihan Rumah Terbaik di Paris</h1>"
                "<p>Paris Housing Dataset · Metode Simple Additive Weighting (SAW) · SCPK 2025/2026</p></div>",
                unsafe_allow_html=True)
 
    cols = st.columns(4)
    for c, (icon, val, lbl) in zip(cols, [
        ("📊", f"{len(df):,}", "Total Data"), ("📐", f"{len(df.columns)}", "Total Kolom"),
        ("🎯", "6", "Kriteria SAW"),           ("🔘", "4", "Filter Fasilitas"),
    ]):
        c.markdown(f"<div class='mcard'><div style='font-size:1.3rem'>{icon}</div>"
                   f"<div class='v'>{val}</div><div class='l'>{lbl}</div></div>", unsafe_allow_html=True)
 
    st.markdown("<div class='shead'>⚖️ Kriteria SAW</div>", unsafe_allow_html=True)
    crit_info = ["Luas bangunan (m²)", "Jumlah total kamar", "Zona lokasi kota (tinggi = prestisius)",
                 "Jumlah pemilik sebelumnya (Cost)", "Ketersediaan kamar tamu (1=ada)", "Harga rumah (Cost)"]
    st.dataframe(pd.DataFrame({"Kode":[f"C{i+1}" for i in range(6)], "Atribut":CRITERIA,
        "Label":CRITERIA_LABELS, "Tipe":CRITERIA_TYPES, "Keterangan":crit_info}),
        use_container_width=True, hide_index=True)
 
    st.markdown("<div class='shead'>🔘 Filter Fasilitas</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Fasilitas":[f"{f['icon']} {f['label']}" for f in FASIL],
        "Kolom":[f["col"] for f in FASIL]}), use_container_width=True, hide_index=True)
 
    st.markdown("<div class='shead'>📄 Dataset</div>", unsafe_allow_html=True)
    n = st.number_input("Tampilkan baris:", 10, len(df), 50, 10)
    st.dataframe(df.head(n), use_container_width=True, height=360)
    st.caption(f"{min(n,len(df)):,} dari {len(df):,} baris ditampilkan.")
 
    st.markdown("<div class='shead'>📈 Statistik Deskriptif</div>", unsafe_allow_html=True)
    st.dataframe(df[CRITERIA].describe().round(2), use_container_width=True)
 
# ── Halaman 2: Hitung SPK ──────────────────────────────────────────────────────
def page_hitung_spk(df):
    st.markdown("<div class='hero'><h1>⚙️ Perhitungan SPK — Metode SAW</h1>"
                "<p>Step 1: Filter fasilitas → Step 2: Bobot kriteria → Step 3: Jalankan SAW</p></div>",
                unsafe_allow_html=True)
 
    # STEP 1 — Filter Fasilitas
    st.markdown("<div class='shead'><span class='pill'>STEP 1</span> &nbsp;Preferensi Fasilitas</div>", unsafe_allow_html=True)
    prefs = {}
    cols = st.columns(4)
    for i, f in enumerate(FASIL):
        with cols[i]:
            st.markdown(f"**{f['icon']} {f['label']}**")
            pilihan = st.selectbox("", ["Tidak Ada Preferensi","✅ Harus Ada","❌ Tidak Perlu"],
                                   key=f"pref_{f['col']}", label_visibility="collapsed")
            prefs[f["col"]] = 1 if "Harus" in pilihan else (0 if "Tidak Perlu" in pilihan else None)
 
    df_f = apply_filter(df.copy(), prefs)
    active = [f"{FASIL[i]['icon']} {FASIL[i]['label']}: {'Ada✅' if v==1 else 'Tdk❌'}"
              for i,(c,v) in enumerate(prefs.items()) if v is not None]
    info_txt = f"Filter aktif: {' | '.join(active)} — **{len(df_f):,} rumah**" if active else f"Semua **{len(df_f):,} rumah** (tanpa filter)"
    st.info(info_txt)
 
    if len(df_f) == 0:
        st.error("⚠️ Tidak ada data yang memenuhi filter. Longgarkan preferensi.")
        return
 
    # STEP 2 — Bobot Kriteria
    st.markdown("---")
    st.markdown("<div class='shead'><span class='pill'>STEP 2</span> &nbsp;Bobot Kriteria (1–10)</div>", unsafe_allow_html=True)
    weights = []
    cl, cr = st.columns(2)
    for i, (col_key, label, ctype, default) in enumerate(zip(CRITERIA, CRITERIA_LABELS, CRITERIA_TYPES, DEFAULT_WEIGHTS)):
        badge_color = "#667eea" if ctype == "Benefit" else "#f64f59"
        badge = f"<span style='background:{badge_color};color:white;border-radius:4px;padding:1px 6px;font-size:0.7rem'>{ctype}</span>"
        with (cl if i % 2 == 0 else cr):
            st.markdown(f"**C{i+1} — {label}** {badge}", unsafe_allow_html=True)
            weights.append(st.slider("", 1, 10, default, key=f"w_{col_key}", label_visibility="collapsed"))
 
    w_norm = np.array(weights, dtype=float) / sum(weights)
    st.dataframe(pd.DataFrame({"Kriteria":CRITERIA_LABELS,"Tipe":CRITERIA_TYPES,
        "Bobot":weights,"Bobot Norm.":[f"{w:.4f}" for w in w_norm],"%":[f"{w*100:.1f}%" for w in w_norm]}),
        use_container_width=True, hide_index=True)
 
    # STEP 3 — Eksekusi
    st.markdown("---")
    st.markdown("<div class='shead'><span class='pill'>STEP 3</span> &nbsp;Eksekusi & Hasil</div>", unsafe_allow_html=True)
    top_n = st.number_input("Top-N ditampilkan:", 5, len(df_f), min(20, len(df_f)), 5)
    show_norm = st.checkbox("Tampilkan Matriks Normalisasi")
 
    _, btn_col, _ = st.columns([1,2,1])
    with btn_col:
        run = st.button("🚀  JALANKAN PERHITUNGAN SAW")
 
    if run:
        with st.spinner("Menghitung skor SAW..."):
            result, norm_df, w_arr = run_saw(df_f, weights)
 
        st.success(f"✅ Selesai! **{len(result):,} rumah** berhasil dirangking.")
 
        if show_norm:
            st.markdown("<div class='shead'>🔢 Matriks Normalisasi (50 baris)</div>", unsafe_allow_html=True)
            nd = norm_df.head(50).copy(); nd.columns = CRITERIA_LABELS; nd.index += 1
            st.dataframe(nd.style.format("{:.4f}"), use_container_width=True, height=280)
 
        st.markdown(f"<div class='shead'>🏆 Hasil Perangkingan SAW — Top {top_n}</div>", unsafe_allow_html=True)
        fasil_cols = [f["col"] for f in FASIL if f["col"] in result.columns]
        disp_cols  = ["Peringkat"] + CRITERIA + fasil_cols + ["Skor SAW"]
        disp_names = ["Peringkat"] + CRITERIA_LABELS + [f["label"] for f in FASIL if f["col"] in result.columns] + ["Skor SAW"]
        rd = result[disp_cols].head(top_n).copy(); rd.columns = disp_names; rd.index = range(1, len(rd)+1)
        st.dataframe(rd.style.background_gradient(subset=["Skor SAW"], cmap="Blues")
                       .format({"Skor SAW":"{:.6f}","Harga (€)":"{:,.0f}","Luas (m²)":"{:.1f}"}),
                     use_container_width=True, height=460)
 
        st.session_state.update({"saw_result":result,"saw_weights":weights,
                                   "w_norm":w_arr,"n_filtered":len(df_f),"active_filters":active})
 
# ── Halaman 3: Visualisasi ─────────────────────────────────────────────────────
def page_visualisasi(df):
    st.markdown("<div class='hero'><h1>📊 Visualisasi Data Analitik</h1>"
                "<p>Distribusi, korelasi, dan perbandingan kriteria dataset Paris Housing</p></div>",
                unsafe_allow_html=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    BG = "#f8f9ff"
 
    # VIZ 1 — Distribusi Harga
    st.markdown("<div class='shead'>📈 1. Distribusi Harga Rumah</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4)); fig.patch.set_facecolor(BG)
    ax[0].hist(df["price"], bins=50, color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax[0].set(title="Histogram Harga", xlabel="Harga (€)", ylabel="Frekuensi"); ax[0].set_facecolor(BG)
    ax[0].ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    sns.kdeplot(data=df, x="price", ax=ax[1], color=PALETTE[1], fill=True, alpha=0.45)
    ax[1].set(title="Density Plot Harga", xlabel="Harga (€)", ylabel="Density"); ax[1].set_facecolor(BG)
    ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 2 — Scatter Luas vs Harga
    st.markdown("<div class='shead'>🔵 2. Luas Bangunan vs Harga</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sc = ax.scatter(df["squareMeters"], df["price"], c=df["cityPartRange"], cmap="plasma", alpha=0.4, s=12, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Zona Kota")
    ax.set(xlabel="Luas (m²)", ylabel="Harga (€)", title="Luas vs Harga — Warna = Zona Kota")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 3 — Heatmap Korelasi
    st.markdown("<div class='shead'>🌡️ 3. Heatmap Korelasi Kriteria SAW</div>", unsafe_allow_html=True)
    corr = df[CRITERIA].corr(); corr.columns = CRITERIA_LABELS; corr.index = CRITERIA_LABELS
    fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor(BG)
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr,bool)), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.5, linecolor="white", annot_kws={"size":9}, square=True)
    ax.set_title("Matriks Korelasi Kriteria SAW", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 4 — Rata-rata Kriteria
    st.markdown("<div class='shead'>📊 4. Rata-Rata Nilai Kriteria SAW</div>", unsafe_allow_html=True)
    means = df[CRITERIA].mean()
    fig, ax = plt.subplots(figsize=(11, 4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.bar(CRITERIA_LABELS, means, color=PALETTE[:6], alpha=0.88, edgecolor="white", width=0.6)
    for b, v in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+means.max()*0.012, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(title="Rata-Rata Nilai Setiap Kriteria SAW", ylabel="Nilai Rata-Rata")
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 5 — Fasilitas
    st.markdown("<div class='shead'>🥧 5. Persentase Fasilitas</div>", unsafe_allow_html=True)
    pcts  = [df[f["col"]].mean()*100 for f in FASIL if f["col"] in df.columns]
    names = [f"{f['icon']} {f['label']}" for f in FASIL if f["col"] in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); fig.patch.set_facecolor(BG)
    axes[0].pie(pcts, labels=names, autopct="%1.1f%%", colors=PALETTE[:len(pcts)],
                startangle=140, wedgeprops={"linewidth":2,"edgecolor":"white"})
    axes[0].set_title("Proporsi Fasilitas"); axes[0].set_facecolor(BG)
    hb = axes[1].barh(range(len(names)), pcts, color=PALETTE[:len(pcts)], alpha=0.85, edgecolor="white")
    axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names)
    axes[1].set(xlabel="Persentase (%)", title="Persentase per Fasilitas"); axes[1].set_facecolor(BG)
    for b, p in zip(hb, pcts):
        axes[1].text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{p:.1f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 6 — Distribusi numPrevOwners
    st.markdown("<div class='shead'>📦 6. Distribusi Pemilik Sebelumnya</div>", unsafe_allow_html=True)
    oc = df["numPrevOwners"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.bar(oc.index.astype(str), oc.values, color=plt.cm.viridis(np.linspace(0.2,0.8,len(oc))), alpha=0.88, edgecolor="white")
    ax.set(xlabel="Jumlah Pemilik Sebelumnya", ylabel="Jumlah Rumah", title="Distribusi numPrevOwners (Kriteria Cost)")
    for x, y in enumerate(oc.values):
        ax.text(x, y+oc.max()*0.01, str(y), ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); st.markdown("---")
 
    # VIZ 7 — Hasil SAW
    if "saw_result" in st.session_state:
        st.markdown("<div class='shead'>🏆 7. Top 15 Rumah Terbaik (Hasil SAW)</div>", unsafe_allow_html=True)
        res = st.session_state["saw_result"].head(15)
        n_f = st.session_state.get("n_filtered", len(res))
        fig, ax = plt.subplots(figsize=(11, 6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15))[::-1]
        bars = ax.barh(range(15), res["Skor SAW"].values, color=colors, alpha=0.9, edgecolor="white")
        ax.set_yticks(range(15)); ax.set_yticklabels([f"#{i+1}" for i in range(15)])
        ax.set(xlabel="Skor SAW", title=f"Top 15 Rumah Terbaik di Paris — SAW\n(dari {n_f:,} rumah setelah filter)")
        ax.invert_yaxis()
        for b, v in zip(bars, res["Skor SAW"].values):
            ax.text(b.get_width()+0.0003, b.get_y()+b.get_height()/2, f"{v:.4f}", va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("💡 Jalankan perhitungan SAW di halaman **⚙️ Hitung SPK** terlebih dahulu.")
 
# ── Halaman 4: Profil ──────────────────────────────────────────────────────────
def page_profil():
    st.markdown("<div class='hero'><h1>👥 Profil Kelompok</h1>"
                "<p>SCPK 2025/2026 · Proyek Akhir Praktikum</p></div>", unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
    c1.markdown("<div class='pcard' style='background:linear-gradient(135deg,#667eea,#764ba2)'>"
                "<h3>👤 Anggota 1</h3><p><b>Nama :</b> [Nama Lengkap 1]</p>"
                "<p><b>NIM &nbsp;:</b> [NIM 1]</p><p><b>Kelas :</b> [Kelas]</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='pcard' style='background:linear-gradient(135deg,#f64f59,#c471ed)'>"
                "<h3>👤 Anggota 2</h3><p><b>Nama :</b> [Nama Lengkap 2]</p>"
                "<p><b>NIM &nbsp;:</b> [NIM 2]</p><p><b>Kelas :</b> [Kelas]</p></div>", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div class='shead'>📋 Informasi Proyek</div>", unsafe_allow_html=True)
    rows = [("🎯 Judul","SPK Pemilihan Rumah Terbaik di Paris"),
            ("📚 Mata Kuliah","Sistem Pengambilan Keputusan (SCPK)"),
            ("📅 Tahun","2025/2026"), ("⚡ Metode","Simple Additive Weighting (SAW)"),
            ("📊 Dataset","Paris Housing Price Prediction (Kaggle)"),
            ("🔗 Sumber","https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction"),
            ("⚖️ Kriteria SAW",", ".join(CRITERIA)),
            ("🔘 Filter Fasilitas",", ".join(f["col"] for f in FASIL)),
            ("🖥️ Framework","Streamlit (Python)"),
            ("📦 Library","Pandas, NumPy, Matplotlib, Seaborn")]
    for lbl, val in rows:
        a, b = st.columns([1, 3])
        a.markdown(f"**{lbl}**")
        b.markdown(f"[{val}]({val})" if val.startswith("http") else val)
 
    st.markdown("---")
    st.markdown("<div class='shead'>📌 Cara Penggunaan</div>", unsafe_allow_html=True)
    st.markdown("""
1. Pastikan **ParisHousing_modified.csv** dan **app.py** berada di satu folder
2. **🏠 Beranda** → lihat dataset dan info kriteria
3. **⚙️ Hitung SPK** → filter fasilitas → atur bobot → klik JALANKAN
4. **📊 Visualisasi** → eksplorasi grafik dan hasil perangkingan
""")
 
# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df   = load_data()
    menu = render_sidebar()
    if "Beranda"    in menu: page_beranda(df)
    elif "Hitung"   in menu: page_hitung_spk(df)
    elif "Visualisa" in menu: page_visualisasi(df)
    elif "Profil"   in menu: page_profil()
 
if __name__ == "__main__":
    main()