'''
2025-10-10
Author: Dan Schumacher
How to run:
  python ./src/method/classification/dtw1nn.py --input_path ./data/datasets/classification/MONSTER/Pedestrian --output_path ./data/generations/dtw1nn/MONSTER/Pedestrian.jsonl --n_yield_rows 1000
'''
import os
import sys
import json
from typing import Tuple

# ensure local imports work
sys.path.append("./src")

import numpy as np
from utils.argparsers import dtw1nn_parser
from utils.logging_utils import MasterLogger
from utils.file_io import yield_jsonl

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics import accuracy_score, f1_score

def set_up() -> tuple:
    args = dtw1nn_parser()
    logger = MasterLogger(log_path="./logs/baseline.log", init=True, clear=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        pass  # clear

    train_file = os.path.join(args.input_path, "train.jsonl")
    test_file  = os.path.join(args.input_path, "test.jsonl")

    n = args.n_yield_rows
    train = yield_jsonl(train_file, n)
    np.random.seed(42); np.random.shuffle(train)
    test  = yield_jsonl(test_file, n)

    print(f"Loaded {len(train)} train & {len(test)} test")
    print(f"Unique labels (train): {len({rec['label'] for rec in train})}")

    X_train = [np.array(rec["series"]) for rec in train]
    y_train = [rec["label"] for rec in train]
    X_test  = [np.array(rec["series"]) for rec in test]
    y_test  = [rec["label"] for rec in test]

    return args, logger, X_train, y_train, X_test, y_test

def main():
    args, logger, X_train, y_train, X_test, y_test = set_up()

    scaler = TimeSeriesScalerMinMax()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)

    with open(args.output_path, "a") as f:
        for pred, gt in zip(preds, y_test):
            f.write(json.dumps({"pred": int(pred), "gt": int(gt)}) + "\n")

    print("Accuracy:", f"{accuracy_score(y_test, preds):.2%}")
    print("Macro F1:", f"{f1_score(y_test, preds, average='macro'):.2%}")

if __name__ == "__main__":
    main()
    print("SCRIPT COMPLETE")
