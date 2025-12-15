import pandas as pd

def rupiah_int(x) -> str:
    try:
        return f"Rp {int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "-"

def table_rekomendasi(var_names, x, as_rupiah: bool = True):
    rows = []
    for i, name in enumerate(var_names):
        harga = float(x[i])
        rows.append({
            "Produk": name,
            "Harga Rekomendasi": rupiah_int(harga) if as_rupiah else int(round(harga)),
        })
    return pd.DataFrame(rows)

def table_ringkas(revenue, entropy_score, penalty, biaya_total):
    feasible = "YA" if float(penalty) <= 1e-6 else "TIDAK"
    return pd.DataFrame({
        "Keterangan": ["Pendapatan (Revenue)", "Entropy Score (lebih kecil lebih stabil)", "Biaya + Target", "Feasible?", "Penalty"],
        "Nilai": [rupiah_int(revenue), f"{float(entropy_score):.6f}", rupiah_int(biaya_total), feasible, f"{float(penalty):.0f}"],
    })
