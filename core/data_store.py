import pandas as pd
from .config import DATA_PATH

def load_history():
    df = pd.read_csv(DATA_PATH)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df

def get_min_date(df):
    return df["tanggal"].min()
