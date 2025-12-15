# core/predictors.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
ASSET_CSV = ROOT / "assets" / "harga_5thn_long_1pasar.csv"

SVM_PKL = ART_DIR / "svm_global.pkl"
REG_PKL = ART_DIR / "reg_global.pkl"

_CACHE: Dict[str, Any] = {}


def _load_once():
    if _CACHE.get("loaded"):
        return

    if not SVM_PKL.exists():
        raise FileNotFoundError(f"Model SVM tidak ditemukan: {SVM_PKL}")
    if not REG_PKL.exists():
        raise FileNotFoundError(f"Model Regresi tidak ditemukan: {REG_PKL}")
    if not ASSET_CSV.exists():
        raise FileNotFoundError(
            f"CSV historis tidak ditemukan: {ASSET_CSV}\n"
            f"Taruh file cleaning kamu di: assets/harga_5thn_long_1pasar.csv"
        )

    svm_art = joblib.load(SVM_PKL)
    reg_art = joblib.load(REG_PKL)

    _CACHE["svm_model"] = svm_art["model"]
    _CACHE["reg_model"] = reg_art["model"]

    _CACHE["feature_cols"] = svm_art.get("feature_cols")
    _CACHE["min_date"] = pd.to_datetime(svm_art.get("min_date"))

    df = pd.read_csv(ASSET_CSV)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")
    df = df.dropna(subset=["komoditas", "kualitas", "tanggal", "harga"])
    df = df[df["harga"] > 0].copy()
    df = df.sort_values(["komoditas", "kualitas", "tanggal"]).reset_index(drop=True)

    _CACHE["hist_df"] = df
    _CACHE["loaded"] = True


def _build_X(komoditas: str, kualitas: str, tanggal: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fitur untuk model global:
    - waktu: tahun, bulan, minggu, hari_ke
    - lag/rolling: lag_1, lag_7, mean_7, mean_14, delta_7
    """
    _load_once()
    tanggal = pd.to_datetime(tanggal)

    df = _CACHE["hist_df"]
    df_prod = df[(df["komoditas"] == komoditas) & (df["kualitas"] == kualitas)].copy()
    if df_prod.empty:
        return None, "Produk tidak ditemukan di data historis (cek komoditas/kualitas)."

    df_hist = df_prod[df_prod["tanggal"] < tanggal].sort_values("tanggal")
    if len(df_hist) < 14:
        return None, "Histori kurang (butuh minimal 14 data sebelum tanggal prediksi)."

    lag_1 = float(df_hist.iloc[-1]["harga"])
    lag_7 = float(df_hist.iloc[-7]["harga"])
    mean_7 = float(df_hist["harga"].tail(7).mean())
    mean_14 = float(df_hist["harga"].tail(14).mean())
    delta_7 = float(lag_1 - lag_7)

    min_date = _CACHE["min_date"]
    if pd.isna(min_date):
        min_date = df["tanggal"].min()

    row = {
        "komoditas": komoditas,
        "kualitas": kualitas,
        "tahun": int(tanggal.year),
        "bulan": int(tanggal.month),
        "minggu": int(tanggal.isocalendar().week),
        "hari_ke": int((tanggal - min_date).days),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "mean_7": mean_7,
        "mean_14": mean_14,
        "delta_7": delta_7,
    }

    X = pd.DataFrame([row])

    feature_cols = _CACHE["feature_cols"]
    if not feature_cols:
        feature_cols = list(X.columns)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    return X, "OK"


def get_predictions(
    komoditas: str,
    kualitas: str,
    tanggal: pd.Timestamp,
    ref_date=None,
) -> Tuple[Optional[float], Optional[float], str, str]:
    try:
        X, status = _build_X(komoditas, kualitas, tanggal)
        if X is None:
            return None, None, f"SVM: {status}", f"Regresi: {status}"

        pred_svm = float(_CACHE["svm_model"].predict(X)[0])
        pred_reg = float(_CACHE["reg_model"].predict(X)[0])

        if (not np.isfinite(pred_svm)) or pred_svm <= 0:
            pred_svm, svm_status = None, "SVM prediksi tidak valid (<=0)."
        else:
            svm_status = "OK"

        if (not np.isfinite(pred_reg)) or pred_reg <= 0:
            pred_reg, reg_status = None, "Regresi prediksi tidak valid (<=0)."
        else:
            reg_status = "OK"

        return pred_svm, pred_reg, svm_status, reg_status

    except Exception as e:
        return None, None, f"SVM error: {e}", f"Regresi error: {e}"


def _shannon_entropy_from_series(prices: np.ndarray, bins: int = 12) -> float:
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    prices = prices[prices > 0]
    if len(prices) < 10:
        return float("nan")

    hist, _ = np.histogram(prices, bins=bins)
    p = hist.astype(float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def get_entropy(
    komoditas: str,
    kualitas: str,
    tanggal: pd.Timestamp,
    window_days: int = 90,
    bins: int = 12,
) -> Tuple[Optional[float], str]:
    """
    Entropy pasar (Shannon entropy) dari distribusi harga historis.
    Ambil window N hari terakhir sebelum tanggal prediksi.
    """
    try:
        _load_once()
        tanggal = pd.to_datetime(tanggal)

        df = _CACHE["hist_df"]
        df_prod = df[(df["komoditas"] == komoditas) & (df["kualitas"] == kualitas)].copy()
        if df_prod.empty:
            return None, "Produk tidak ditemukan di historis."

        df_hist = df_prod[df_prod["tanggal"] < tanggal].sort_values("tanggal")
        if df_hist.empty:
            return None, "Tidak ada histori sebelum tanggal prediksi."

        start = df_hist["tanggal"].max() - pd.Timedelta(days=int(window_days))
        df_win = df_hist[df_hist["tanggal"] >= start]

        ent = _shannon_entropy_from_series(df_win["harga"].to_numpy(), bins=bins)
        if not np.isfinite(ent):
            ent = _shannon_entropy_from_series(df_hist["harga"].to_numpy(), bins=bins)

        if not np.isfinite(ent):
            return None, "Histori terlalu sedikit untuk entropy."
        return float(ent), "OK"

    except Exception as e:
        return None, f"Entropy error: {e}"


def debug_available_models() -> Dict[str, Any]:
    return {
        "artifacts_dir": str(ART_DIR),
        "svm_file": SVM_PKL.name,
        "reg_file": REG_PKL.name,
        "svm_exists": SVM_PKL.exists(),
        "reg_exists": REG_PKL.exists(),
        "assets_csv": str(ASSET_CSV),
        "assets_csv_exists": ASSET_CSV.exists(),
    }
