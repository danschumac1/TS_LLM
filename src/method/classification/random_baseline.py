# ./src/method/classification/random_baseline.py
"""
2025-10-10
Author: Dan Schumacher
How to run:
  python ./src/method/classification/random_baseline.py \
    --input_path ./data/datasets/classification/MONSTER/Pedestrian \
    --output_path ./data/generations/random/MONSTER/Pedestrian.jsonl \
    --n_yield_rows 1000 \
    --mode prior --seed 42
"""
import os, sys, json, argparse
import random
from collections import Counter
import numpy as np
sys.path.append("./src")

from utils.argparsers import random_class_parser
from utils.file_io import yield_jsonl
from sklearn.metrics import accuracy_score, f1_score

def main():
    args = random_class_parser()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w"): pass  # clear

    train_file = os.path.join(args.input_path, "train.jsonl")
    test_file  = os.path.join(args.input_path, "test.jsonl")

    train = yield_jsonl(train_file, args.n_yield_rows)
    test  = yield_jsonl(test_file, args.n_yield_rows)

    y_train = [rec["label"] for rec in train]
    y_test  = [rec["label"] for rec in test]

    labels = sorted(set(y_train))  # label vocabulary from train
    if not labels:
        raise ValueError("No labels found in train split.")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Build sampler
    if args.mode == "uniform":
        def sample(): return rng.choice(labels)
    elif args.mode == "prior":
        counts = Counter(y_train)
        probs = np.array([counts[l] for l in labels], dtype=float)
        probs = probs / probs.sum()
        def sample(): return np.random.choice(labels, p=probs)
    else:  # majority
        majority = Counter(y_train).most_common(1)[0][0]
        def sample(): return majority

    preds = [int(sample()) for _ in y_test]

    # Write predictions
    with open(args.output_path, "a") as f:
        for pred, gt in zip(preds, y_test):
            f.write(json.dumps({"pred": int(pred), "gt": int(gt)}) + "\n")

    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"Mode: {args.mode}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Macro F1: {f1m:.2%}")

if __name__ == "__main__":
    main()
    print("SCRIPT COMPLETE")
