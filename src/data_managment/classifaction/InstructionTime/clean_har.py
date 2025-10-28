"""
2025-10-23
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/clean_har.py

har (UCI Human Activity Recognition)
  - feature vectors: 561 dims
  - num classes: 6
  - num samples: 10,299
  - source: train/X_train.txt, y_train.txt, test/X_test.txt, y_test.txt
"""

import os
import numpy as np

RAW = "data/datasets/_raw_data/InstructionTime/har"
OUT = "data/datasets/classification/har"
DTYPE = np.float32
SEED = 1337

def main():
    os.makedirs(OUT, exist_ok=True)

    Xtr = np.loadtxt(os.path.join(RAW, "train", "X_train.txt"), dtype=DTYPE)
    ytr = np.loadtxt(os.path.join(RAW, "train", "y_train.txt"), dtype=np.int64)
    Xte = np.loadtxt(os.path.join(RAW, "test", "X_test.txt"), dtype=DTYPE)
    yte = np.loadtxt(os.path.join(RAW, "test", "y_test.txt"), dtype=np.int64)

    # Normalize labels 0–5
    ytr -= ytr.min()
    yte -= yte.min()

    # Shuffle training
    rng = np.random.default_rng(SEED)
    p = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[p], ytr[p]

    np.save(os.path.join(OUT, "train.npy"), Xtr)
    np.save(os.path.join(OUT, "train_labels.npy"), ytr)
    np.save(os.path.join(OUT, "test.npy"), Xte)
    np.save(os.path.join(OUT, "test_labels.npy"), yte)

    print(f"✅ train {Xtr.shape} test {Xte.shape}")

if __name__ == "__main__":
    main()
