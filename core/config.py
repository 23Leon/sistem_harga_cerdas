from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "assets" / "harga_5thn_long_1pasar.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"

KONFIG = {
    "Cabai Merah": [
        "Cabai Merah Besar",
        "Cabai Merah Keriting",
    ],
    "Cabai Rawit": [
        "Cabai Rawit Hijau",
        "Cabai Rawit Merah",
    ],
}

SVM_GLOBAL_MODEL = "svm_global.pkl"
REG_GLOBAL_MODEL = "reg_global.pkl"

SVM_MODEL_MAP = {pair: SVM_GLOBAL_MODEL for kom, kuls in KONFIG.items() for pair in [(kom, k) for k in kuls]}
REG_MODEL_MAP = {pair: REG_GLOBAL_MODEL for kom, kuls in KONFIG.items() for pair in [(kom, k) for k in kuls]}
