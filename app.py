import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictors import get_predictions, get_entropy, debug_available_models
from core.optimizer_pso_entropy import pso_optimize_entropy_aprp
from core.tables_userfriendly import table_rekomendasi, table_ringkas


def rupiah_int(x) -> str:
    try:
        return f"Rp {int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "-"

def auto_scale_price(pred_svm, pred_reg, hist_ranges):
    candidates = []
    for p in (pred_svm, pred_reg):
        if p is not None and np.isfinite(p) and p > 0:
            candidates.append(float(p))
    for (mn, mx) in hist_ranges:
        if mn is not None and mx is not None:
            candidates.append(float(mx))
    if not candidates:
        return 1
    return 1000 if float(np.median(candidates)) < 1000 else 1

def mode_from_trust(k: float) -> tuple[str, str]:
    if k < (1/3):
        return "RENDAH", "Lebih mengandalkan data harga lama."
    elif k <= (2/3):
        return "SEDANG", "Mengandalkan perkiraan dari tren (regresi)."
    else:
        return "TINGGI", "Mengandalkan perkiraan dari model prediksi (SVM)."

def kondisi_pasar_text(ent: float | None) -> str:
    if ent is None:
        return "Belum terbaca"
    if ent < 1.0:
        return "Cenderung stabil"
    if ent < 1.6:
        return "Agak berubah-ubah"
    return "Sering naik-turun"

def clamp_minmax(lo, hi):
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def ensure_state():
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "selected" not in st.session_state:
        st.session_state.selected = []
    if "hist_ranges" not in st.session_state:
        st.session_state.hist_ranges = []
    if "pred_svm_list" not in st.session_state:
        st.session_state.pred_svm_list = []
    if "pred_reg_list" not in st.session_state:
        st.session_state.pred_reg_list = []
    if "entropy_list" not in st.session_state:
        st.session_state.entropy_list = []
    if "hhat" not in st.session_state:
        st.session_state.hhat = []
    if "hhat_src" not in st.session_state:
        st.session_state.hhat_src = []
    if "inputs_opt" not in st.session_state:
        st.session_state.inputs_opt = {}  
    if "result" not in st.session_state:
        st.session_state.result = None  

st.set_page_config(page_title="Harga Cerdas Pangan", page_icon="🌶️", layout="wide")
ensure_state()

st.title("🌶️ Harga Cerdas Pangan")

with st.sidebar:
    st.header("⚙️ Pengaturan")

    tanggal = st.date_input("Tanggal rencana jual", value=pd.Timestamp.today().date())
    tanggal_ts = pd.to_datetime(tanggal)

    st.subheader("Kondisi Pasar")
    window_days = st.slider("Data berapa hari terakhir?", 30, 365, 90, 15)
    bins = st.slider("Tingkat ketelitian analisis", 6, 30, 12, 1)

    st.subheader("Tingkat Kepercayaan")
    k_ts = st.slider("Seberapa kamu percaya hasil perkiraan harga?", 0.0, 1.0, 0.70, 0.05)
    mode, mode_desc = mode_from_trust(float(k_ts))
    st.info(f"Mode: **{mode}**\n\n{mode_desc}")

    st.subheader("Rekomendasi Harga")
    w_entropy = st.slider("Utamakan harga yang stabil", 0.0, 10.0, 1.0, 0.1)
    st.caption("Kalau dinaikkan, sistem cenderung memilih harga yang lebih stabil.")

    st.subheader("Mesin Rekomendasi")
    n_particles = st.slider("Jumlah kandidat yang dicoba", 30, 300, 80, 10)
    n_iter = st.slider("Berapa kali perbaikan", 50, 800, 200, 10)

    st.divider()
    if st.button("🔁 Reset & mulai dari awal"):
        for k in list(st.session_state.keys()):
            if k not in ("step",):
                pass
        st.session_state.step = 1
        st.session_state.selected = []
        st.session_state.hist_ranges = []
        st.session_state.pred_svm_list = []
        st.session_state.pred_reg_list = []
        st.session_state.entropy_list = []
        st.session_state.hhat = []
        st.session_state.hhat_src = []
        st.session_state.inputs_opt = {}
        st.session_state.result = None
        st.rerun()


OPTIONS = [
    ("Cabai Merah", "Cabai Merah Besar"),
    ("Cabai Merah", "Cabai Merah Keriting"),
    ("Cabai Rawit", "Cabai Rawit Hijau"),
    ("Cabai Rawit", "Cabai Rawit Merah"),
]

if st.session_state.step == 1:
    st.header("Mulai dari sini")

    with st.container(border=True):
        st.write("👇 Pilih produk yang ingin kamu cek")

        selected = st.multiselect(
            "Pilih jenis cabai (boleh 1 sampai 4)",
            options=OPTIONS,
            default=st.session_state.selected if st.session_state.selected else [OPTIONS[0]],
            format_func=lambda t: f"{t[0]} | {t[1]}",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("➡️ Lanjut", type="primary"):
                if len(selected) < 1:
                    st.warning("Pilih minimal 1 produk dulu ya.")
                else:
                    st.session_state.selected = selected
                    st.session_state.step = 2
                    st.rerun()
    

    st.stop()
if st.session_state.step == 2:
    selected = st.session_state.selected
    n_var = len(selected)
    var_names = [f"Produk {i+1}" for i in range(n_var)]

    st.header("Lihat perkiraan harga")

    with st.container(border=True):

        pred_svm_list, pred_reg_list, entropy_list = [], [], []
        hist_ranges = []
        rows_summary = []

        for i, (kom, kul) in enumerate(selected):
            st.markdown(f"### {var_names[i]} — {kom} | {kul}")

            cA, cB = st.columns(2)
            with cA:
                mn = st.number_input(
                    f"Harga terendah dari data lama ({var_names[i]})",
                    min_value=0,
                    value=1000,
                    step=500,
                    format="%d",
                    key=f"mn_{i}",
                )
            with cB:
                mx = st.number_input(
                    f"Harga tertinggi dari data lama ({var_names[i]})",
                    min_value=0,
                    value=5000,
                    step=500,
                    format="%d",
                    key=f"mx_{i}",
                )
            hist_ranges.append((int(mn), int(mx)))

            pred_svm, pred_reg, svm_status, reg_status = get_predictions(kom, kul, tanggal_ts)
            scale = auto_scale_price(pred_svm, pred_reg, hist_ranges=[(mn, mx)])
            show_svm = None if pred_svm is None else float(pred_svm) * scale
            show_reg = None if pred_reg is None else float(pred_reg) * scale

            ent, ent_status = get_entropy(kom, kul, tanggal_ts, window_days=int(window_days), bins=int(bins))
            ent_val = float(ent) if ent is not None else None

            pred_svm_list.append(show_svm)
            pred_reg_list.append(show_reg)
            entropy_list.append(float(ent_val) if ent_val is not None else 0.0)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Perkiraan (SVM)", rupiah_int(show_svm) if show_svm is not None else "—")
                if show_svm is None:
                    st.caption(f"Catatan: {svm_status}")
            with m2:
                st.metric("Perkiraan (Regresi)", rupiah_int(show_reg) if show_reg is not None else "—")
                if show_reg is None:
                    st.caption(f"Catatan: {reg_status}")
            with m3:
                st.metric("Kondisi Pasar", kondisi_pasar_text(ent_val))
                if ent_val is not None:
                    st.caption(f"Skor: {ent_val:.3f}")
                else:
                    st.caption(f"Catatan: {ent_status}")

            rows_summary.append({
                "Produk": var_names[i],
                "Komoditas": kom,
                "Kualitas": kul,
                "Perkiraan (SVM)": "-" if show_svm is None else rupiah_int(show_svm),
                "Perkiraan (Regresi)": "-" if show_reg is None else rupiah_int(show_reg),
                "Kondisi Pasar (skor)": "-" if ent_val is None else f"{ent_val:.3f}",
                "Rentang data lama": f"{rupiah_int(mn)} – {rupiah_int(mx)}",
            })

        st.markdown("#### Ringkasan")
        st.dataframe(pd.DataFrame(rows_summary), use_container_width=True)

        st.session_state.hist_ranges = hist_ranges
        st.session_state.pred_svm_list = pred_svm_list
        st.session_state.pred_reg_list = pred_reg_list
        st.session_state.entropy_list = entropy_list

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("⬅️ Kembali"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("➡️ Lanjut", type="primary"):
                st.session_state.step = 3
                st.rerun()
    st.stop()

if st.session_state.step == 3:
    selected = st.session_state.selected
    hist_ranges = st.session_state.hist_ranges
    pred_svm_list = st.session_state.pred_svm_list
    pred_reg_list = st.session_state.pred_reg_list
    entropy_list = st.session_state.entropy_list

    n_var = len(selected)
    var_names = [f"Produk {i+1}" for i in range(n_var)]

    st.header("Atur target & batas harga")

    hhat, hhat_src = [], []
    for i in range(n_var):
        mn, mx = hist_ranges[i]
        svm_val = pred_svm_list[i]
        reg_val = pred_reg_list[i]

        if mode == "RENDAH":
            base = (mn + mx) / 2
            hhat.append(float(base))
            hhat_src.append("Data harga lama (tengah rentang)")
        elif mode == "SEDANG":
            if reg_val is not None:
                hhat.append(float(reg_val))
                hhat_src.append("Perkiraan tren (regresi)")
            else:
                base = (mn + mx) / 2
                hhat.append(float(base))
                hhat_src.append("Tren tidak tersedia → pakai data lama")
        else:
            if svm_val is not None:
                hhat.append(float(svm_val))
                hhat_src.append("Model prediksi (SVM)")
            else:
                base = (mn + mx) / 2
                hhat.append(float(base))
                hhat_src.append("Model tidak tersedia → pakai data lama")

    st.session_state.hhat = hhat
    st.session_state.hhat_src = hhat_src

    with st.container(border=True):
        st.write("📌 **Harga pegangan** (patokan yang dipakai sebelum bikin rekomendasi)")
        df_patokan = pd.DataFrame({
            "Produk": var_names,
            "Patokan diambil dari": hhat_src,
            "Harga pegangan": [rupiah_int(v) for v in hhat],
            "Kondisi pasar": [kondisi_pasar_text(None if entropy_list[i] == 0 else entropy_list[i]) for i in range(n_var)],
        })
        st.table(df_patokan)

    st.divider()

    with st.container(border=True):
        st.write("💰 **Perkiraan jual & target**")

        saved_inputs = st.session_state.inputs_opt or {}

        q_vals = []
        for i in range(n_var):
            default_q = int(saved_inputs.get("q_vals", [1]*n_var)[i]) if "q_vals" in saved_inputs else 1
            q_vals.append(
                st.number_input(
                    f"Perkiraan jumlah jual ({var_names[i]})",
                    min_value=1,
                    value=int(default_q),
                    step=1,
                    format="%d",
                    key=f"q_step3_{i}",
                )
            )

        TC = st.number_input(
            "Perkiraan biaya",
            min_value=0,
            value=int(saved_inputs.get("TC", 0)),
            step=100000,
            format="%d",
            key="TC_step3",
        )
        Profit = st.number_input(
            "Target keuntungan",
            min_value=0,
            value=int(saved_inputs.get("Profit", 0)),
            step=100000,
            format="%d",
            key="Profit_step3",
        )
        r = st.slider(
            "Tingkat kehati-hatian",
            0.0,
            0.5,
            float(saved_inputs.get("r", 0.03)),
            0.01,
            key="r_step3",
        )

        biaya_total = float(TC + Profit)

    st.divider()

    with st.container(border=True):
        st.write("🧾 **Batas harga (boleh diubah)**")

        h_bounds, H_bounds, e_pred = [], [], []
        saved_bounds = saved_inputs.get("bounds", None)

        for i in range(n_var):
            base = float(hhat[i])
            default_lo = max(0.0, base * 0.85)
            default_hi = max(default_lo, base * 1.15)
            if saved_bounds and i < len(saved_bounds):
                default_lo = saved_bounds[i].get("lo", default_lo)
                default_hi = saved_bounds[i].get("hi", default_hi)
                default_e = saved_bounds[i].get("e", 0.25)
            else:
                default_e = 0.25

            st.markdown(f"**{var_names[i]}**")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                lo = st.number_input(
                    "Harga terendah",
                    min_value=0,
                    value=int(round(default_lo)),
                    step=100,
                    format="%d",
                    key=f"lo_step3_{i}",
                )
            with c2:
                hi = st.number_input(
                    "Harga tertinggi",
                    min_value=0,
                    value=int(round(default_hi)),
                    step=100,
                    format="%d",
                    key=f"hi_step3_{i}",
                )
            with c3:
                e = st.number_input(
                    "Toleransi perubahan",
                    min_value=0.0,
                    max_value=0.50,
                    value=float(default_e),
                    step=0.01,
                    key=f"e_step3_{i}",
                )
                st.caption("Contoh: 0.25 = toleransi sekitar 25% dari patokan.")

            lo, hi = clamp_minmax(lo, hi)
            h_bounds.append(float(lo))
            H_bounds.append(float(hi))
            e_pred.append(float(e))

    
        rev_max = float(np.sum(np.asarray(q_vals, dtype=float) * np.asarray(H_bounds, dtype=float)))
        rev_max_safe = rev_max * (1.0 - float(r))
        st.info(
            f"🔎 Dengan batas harga & jumlah jual yang kamu isi, perkiraan uang masuk maksimal sekitar **{rupiah_int(rev_max)}** "
            f"(lebih aman sekitar **{rupiah_int(rev_max_safe)}**). Target kamu: **{rupiah_int(biaya_total)}**."
        )

    
    st.session_state.inputs_opt = {
        "q_vals": [int(v) for v in q_vals],
        "TC": int(TC),
        "Profit": int(Profit),
        "r": float(r),
        "bounds": [{"lo": h_bounds[i], "hi": H_bounds[i], "e": e_pred[i]} for i in range(n_var)],
    }

    st.divider()

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("⬅️ Kembali"):
            st.session_state.step = 2
            st.rerun()
    with cB:
        run = st.button("🚀 Buat rekomendasi harga", type="primary")
    
    if not run:
        st.stop()

    
    errs = []
    for i in range(n_var):
        if H_bounds[i] < h_bounds[i]:
            errs.append(f"{var_names[i]}: harga tertinggi lebih kecil dari terendah.")
    if errs:
        st.error("❌ Ada input yang perlu diperbaiki:\n- " + "\n- ".join(errs))
        st.stop()

    params = {
        "n_var": n_var,
        "q": [float(v) for v in q_vals],
        "h": [float(v) for v in h_bounds],
        "H": [float(v) for v in H_bounds],
        "TC": float(TC),
        "Profit": float(Profit),
        "r": float(r),
        "hhat": [float(v) for v in hhat],
        "e_pred": [float(v) for v in e_pred],
        "entropy": [float(v) for v in entropy_list],
        "w_entropy": float(w_entropy),
    }

    with st.spinner("Sedang menghitung rekomendasi..."):
        best_x, revenue, ent_score, penalty = pso_optimize_entropy_aprp(
            params,
            n_particles=int(n_particles),
            n_iter=int(n_iter),
            seed=1,
        )

    st.session_state.result = {
        "best_x": best_x,
        "revenue": float(revenue),
        "ent_score": float(ent_score),
        "penalty": float(penalty),
        "biaya_total": float(TC + Profit),
    }
    st.session_state.step = 4
    st.rerun()

if st.session_state.step == 4:
    selected = st.session_state.selected
    res = st.session_state.result
    n_var = len(selected)
    var_names = [f"Produk {i+1}" for i in range(n_var)]

    st.header("Hasil rekomendasi")

    if not res:
        st.warning("Hasil belum ada. Silakan buat rekomendasi dulu.")
        if st.button("⬅️ Kembali"):
            st.session_state.step = 3
            st.rerun()
        st.stop()

    best_x = res["best_x"]
    revenue = res["revenue"]
    ent_score = res["ent_score"]
    penalty = res["penalty"]
    biaya_total = res["biaya_total"]

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Perkiraan uang masuk", rupiah_int(revenue))
        with m2:
            st.metric("Skor kestabilan", f"{float(ent_score):.6f}")
        with m3:
            st.metric("Status", "AMAN" if penalty <= 1e-6 else "PERLU DISESUAIKAN")

        if penalty > 1e-6:
            st.warning("⚠️ Rekomendasi sudah keluar, tapi masih perlu disesuaikan.")
            st.caption(
                "Coba salah satu:\n"
                "- Naikkan toleransi perubahan\n"
                "- Lebarkan batas harga\n"
                "- Kurangi target atau tambah jumlah jual\n"
                "- Turunkan tingkat kehati-hatian"
            )
        else:
            st.success("✅ Rekomendasi Harga.")

    st.divider()

    st.subheader("Rekomendasi harga")
    labels = [f"{var_names[i]} — {selected[i][0]} | {selected[i][1]}" for i in range(n_var)]
    st.table(table_rekomendasi(labels, best_x, as_rupiah=True))

    st.subheader("Ringkasan")
    
    try:
        st.table(table_ringkas(revenue, ent_score, penalty, biaya_total))
    except TypeError:
        
        st.table(table_ringkas(revenue, ent_score, revenue, ent_score, biaya_total))

    with st.expander("Detail (opsional)"):
        df_detail = pd.DataFrame({
            "Produk": labels,
            "Harga rekomendasi": [rupiah_int(v) for v in best_x],
        })
        st.dataframe(df_detail, use_container_width=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("⬅️ Kembali"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("🔁 Coba lagi dari awal"):
            st.session_state.step = 1
            st.session_state.result = None
            st.rerun()

    st.stop()
