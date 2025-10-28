"""
2025-10-22
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/clean_cpu.py
   
    cpu
    univariate
    series length: 720
    num classes: 2
    num samples: 500
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
BASE_PATH = "data/datasets/_raw_data/InstructionTime/cpu/"
OUT_DIR   = "data/datasets/classification/cpu/"
SHUFFLE_TRAIN = True       # Shuffle training set for randomness
SHUFFLE_TEST  = False      # Usually leave test ordered
SEED = 1337                # Fixed seed for reproducibility

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(BASE_PATH, "Computers")

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
    assert X_tr.shape[1] == 720 and X_te.shape[1] == 720, "Series length mismatch!"
    print("Done. Shapes:", X_tr.shape, X_te.shape)

if __name__ == "__main__":
    main()
