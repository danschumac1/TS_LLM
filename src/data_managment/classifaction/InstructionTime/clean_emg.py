"""
2025-10-23
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/clean_emg.py

EMG (Electromyography)
  - univariate
  - series length: 1500
  - num classes: 3  (healthy=0, myopathy=1, neuropathy=2)
  - each class file is one long signal -> we segment into 1500-sample windows
  - split: stratified 85/15 over those windows
"""

import sys; sys.path.append("./src/")  # for utils/
import os
import numpy as np
from typing import Dict, List
from utils.file_io import save_split, stratified_train_test_split

# ----------------------------- CONFIG --------------------------------
RAW_DIR   = "data/datasets/_raw_data/InstructionTime/emg"
OUT_DIR   = "data/datasets/classification/emg"
SERIES_LEN = 1500
DTYPE      = np.float32
TRAIN_FRAC = 0.85
SEED       = 1337
SHUFFLE_WITHIN_SPLITS = True

# Optional per-window normalization
PEAK_NORMALIZE   = False
ZSCORE_NORMALIZE = False

CLASS_FILES: Dict[str, str] = {
    "healthy":    "emg_healthy.txt",
    "myopathy":   "emg_myopathy.txt",
    "neuropathy": "emg_neuropathy.txt",
}
CLASS_LABELS: Dict[str, int] = {"healthy": 0, "myopathy": 1, "neuropathy": 2}

# --------------------------- IO HELPERS ------------------------------
def _decode_lines(path: str):
    # robust text decoding
    try:
        with open(path, "r", encoding="utf-8") as f:
            yield from f; return
    except UnicodeDecodeError:
        pass
    try:
        with open(path, "r", encoding="latin-1") as f:
            yield from f; return
    except UnicodeDecodeError:
        pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        yield from f

def read_time_value_txt(path: str) -> np.ndarray:
    vals: List[float] = []
    for raw in _decode_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            vals.append(float(parts[1]))
        except ValueError:
            continue
    if not vals:
        raise RuntimeError(f"No numeric values parsed from {path}")
    return np.asarray(vals, dtype=DTYPE)

def windows_1d(x: np.ndarray, L: int, stride: int) -> np.ndarray:
    x = x.ravel()
    if len(x) < L: return np.empty((0, L), dtype=x.dtype)
    n = (len(x) - L) // stride + 1
    out = np.empty((n, L), dtype=x.dtype)
    for i in range(n):
        s = i * stride
        out[i] = x[s:s+L]
    return out

# ------------------------------- MAIN --------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_X, all_y = [], []
    print("Loading class files and slicing into 1500-sample windows ...")
    for cname, fname in CLASS_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        x_long = read_time_value_txt(path)
        W = windows_1d(x_long, L=SERIES_LEN, stride=SERIES_LEN)  # non-overlapping
        if PEAK_NORMALIZE or ZSCORE_NORMALIZE:
            for i in range(W.shape[0]):
                w = W[i]
                if PEAK_NORMALIZE:
                    peak = float(np.max(np.abs(w))) + 1e-12
                    w = w / peak
                if ZSCORE_NORMALIZE:
                    mu = float(w.mean()); sd = float(w.std() + 1e-8)
                    w = (w - mu) / sd
                W[i] = w
        y = np.full((W.shape[0],), CLASS_LABELS[cname], dtype=np.int64)
        all_X.append(W)
        all_y.append(y)
        print(f"  {cname:<12} -> {W.shape[0]} windows of length {SERIES_LEN} (from {len(x_long)} samples)")

    X_all = np.vstack(all_X).astype(DTYPE)
    y_all = np.concatenate(all_y).astype(np.int64)
    print(f"Total samples (windows): {len(y_all)} | X shape: {X_all.shape}")

    # stratified 85/15
    tr_idx, te_idx = stratified_train_test_split(y_all, TRAIN_FRAC, SEED)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    if SHUFFLE_WITHIN_SPLITS:
        rng = np.random.default_rng(SEED)
        p = rng.permutation(len(y_tr)); X_tr, y_tr = X_tr[p], y_tr[p]
        p = rng.permutation(len(y_te)); X_te, y_te = X_te[p], y_te[p]

    print("Saving arrays ...")
    save_split(os.path.join(OUT_DIR, "train"), X_tr, y_tr, DTYPE)
    save_split(os.path.join(OUT_DIR, "test"),  X_te, y_te, DTYPE)

    # sanity
    assert X_tr.shape[1] == SERIES_LEN and X_te.shape[1] == SERIES_LEN
    print("✅ Done. train", X_tr.shape, "test", X_te.shape)

if __name__ == "__main__":
    main()
