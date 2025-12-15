import numpy as np

def get_price_range(df_hist, komoditas, kualitas, q_low=0.10, q_high=0.90):
    sub = df_hist[(df_hist["komoditas"] == komoditas) & (df_hist["kualitas"] == kualitas)].copy()
    if len(sub) == 0:
        raise ValueError("Data historis kosong untuk pilihan ini.")
    low = float(sub["harga"].quantile(q_low))
    high = float(sub["harga"].quantile(q_high))
    return low, high

def entropy_from_returns(df_hist, komoditas, kualitas, bins=20):
    sub = df_hist[(df_hist["komoditas"] == komoditas) & (df_hist["kualitas"] == kualitas)].sort_values("tanggal")
    r = sub["harga"].pct_change().dropna()
    if len(r) < 10:
        return 0.0

    hist, _ = np.histogram(r, bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)))
    return H

def categorize_entropy(H, low_th=0.8, high_th=1.2):
    
    if H < low_th:
        return "Rendah (stabil)"
    if H < high_th:
        return "Sedang"
    return "Tinggi (volatil)"
