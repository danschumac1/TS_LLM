"""
2025-10-23
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/clean_whale.py

whale (rcw) (Cornell Whale Challenge)
  - univariate
  - series length: 4000
  - num classes: 2
  - train: AIFFs with labels in train.csv
  - split: stratified 85/15 into train/test (from TRAIN only)
"""
import sys; sys.path.append("./src/")  # noqa
import os
from typing import List, Tuple, Optional, Dict
import numpy as np

from utils.file_io import (
    list_aiff_files, 
    load_aiff_as_row, 
    load_labels_csv, 
    save_split, 
    stem, 
    stratified_train_test_split
    )

# ----------------------------- CONFIG --------------------------------
RAW_ROOT   = "data/datasets/_raw_data/InstructionTime/whale"
TRAIN_DIR  = os.path.join(RAW_ROOT, "train")     # AIFFs here
TRAIN_CSV  = os.path.join(RAW_ROOT, "train.csv") # filename,label

OUT_DIR    = "data/datasets/classification/whale"
TARGET_LEN = 4000
DTYPE      = np.float32

TRAIN_FRAC = 0.85     # stratified 85/15 split
SEED       = 1337
SHUFFLE_WITHIN_SPLITS = True


# ------------------------------- MAIN --------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load labels ---
    print("Reading labels CSV ...")
    label_map = load_labels_csv(TRAIN_CSV)  # stem -> {0,1}

    # --- Scan TRAIN AIFFs ---
    print("Scanning TRAIN AIFFs ...")
    train_files = list_aiff_files(TRAIN_DIR)
    if not train_files:
        raise RuntimeError(f"No AIFF files found in {TRAIN_DIR}")

    X_rows: List[np.ndarray] = []
    y_list: List[int] = []
    missing_label = 0

    for p in train_files:
        st = stem(p)
        y = label_map.get(st)
        if y is None:
            missing_label += 1
            continue  # skip files missing labels
        try:
            row = load_aiff_as_row(p, dtype=DTYPE, target_len=TARGET_LEN)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {p} ({e})")
            continue
        X_rows.append(row)
        y_list.append(int(y))

    if not X_rows:
        raise RuntimeError("No labeled samples loaded from TRAIN.")
    if missing_label:
        print(f"[INFO] Skipped {missing_label} files not present in train.csv.")

    X_all = np.vstack(X_rows).astype(DTYPE)
    y_all = np.asarray(y_list, dtype=np.int64)

    # --- Stratified 85/15 split (TRAIN→train/test) ---
    print("Stratified 85/15 split from TRAIN ...")
    tr_idx, te_idx = stratified_train_test_split(y_all, TRAIN_FRAC, SEED)

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    if SHUFFLE_WITHIN_SPLITS:
        rng = np.random.default_rng(SEED)
        p = rng.permutation(len(y_tr)); X_tr, y_tr = X_tr[p], y_tr[p]
        p = rng.permutation(len(y_te)); X_te, y_te = X_te[p], y_te[p]

    # --- Save ---
    print("Saving ...")
    save_split(os.path.join(OUT_DIR, "train"), X_tr, y_tr, DTYPE)
    save_split(os.path.join(OUT_DIR, "test"),  X_te, y_te, DTYPE)

    # --- Sanity ---
    assert X_tr.shape[1] == TARGET_LEN and X_te.shape[1] == TARGET_LEN, "Length mismatch."
    print(f"Done. train {X_tr.shape}, test {X_te.shape}")

if __name__ == "__main__":
    main()
