"""
2025-10-22
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/clean_tee.py

    tee (Lightning / TEE)
    univariate
    series length: 319
    num classes: 7
    num samples: 143
    -------
    train: arff, ts, and txt file
    test: arff, ts, and txt file
"""


import os
import sys; sys.path.append("./src/")
from utils.file_io import read_txt_file, save_to_npy_pair, shuffle_in_unison

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_PATH = "data/datasets/_raw_data/InstructionTime/lightning/"
OUT_DIR   = "data/datasets/classification/lightning/"
SERIES_LEN = 319            # Lightning7 length
SHUFFLE_TRAIN = True
SHUFFLE_TEST  = False
SEED = 1337

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(BASE_PATH, "Lightning7")

    print("Loading data ...")
    X_tr, y_tr = read_txt_file(base + "_TRAIN.txt")
    X_te, y_te = read_txt_file(base + "_TEST.txt")

    if SHUFFLE_TRAIN:
        X_tr, y_tr = shuffle_in_unison(X_tr, y_tr, SEED)
        print(f"Shuffled training set ({len(y_tr)} samples)")
    if SHUFFLE_TEST:
        X_te, y_te = shuffle_in_unison(X_te, y_te, SEED)
        print(f"Shuffled test set ({len(y_te)} samples)")

    print("Saving arrays ...")
    save_to_npy_pair(X_tr, y_tr, os.path.join(OUT_DIR, "train"))
    save_to_npy_pair(X_te, y_te, os.path.join(OUT_DIR, "test"))

    # sanity check
    assert X_tr.shape[1] == SERIES_LEN and X_te.shape[1] == SERIES_LEN, (
        f"Series length mismatch! expected {SERIES_LEN}, "
        f"got {X_tr.shape[1]} (train) and {X_te.shape[1]} (test)"
    )
    print("Done. Shapes:", X_tr.shape, X_te.shape)

if __name__ == "__main__":
    main()
