"""
2025-09-20
Author: Dan Schumacher
Non-LLM classical baseline: kNN (DTW) for time series classification.
- Compatible with your bash framework & JSONL output writer.
- Uses tslearn for speed (parallel, banded DTW).
How to run:
    see ./bin/knn_classifier.sh
"""

import argparse, os, sys
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Fast DTW kNN
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Your utils (as in your framework)
from utils.file_io import load_jsonl
from utils.data_utils import save_output
from utils.logging_utils import MasterLogger


# -------------------- helpers --------------------

def _sort_key_dim(k: str) -> int:
    try:
        return int(str(k).split("_")[1])
    except Exception:
        return 10**9

def ts_dict_to_np(ts: Any) -> np.ndarray:
    """row['ts'] -> (T, D). Univariate -> D=1."""
    if isinstance(ts, dict):
        dims = [np.asarray(ts[d], dtype="float32") for d in sorted(ts.keys(), key=_sort_key_dim)]
        return np.stack(dims, axis=1)  # (T, D)
    else:
        return np.asarray(ts, dtype="float32")[:, None]

def downsample_td(arr: np.ndarray, step: int) -> np.ndarray:
    if step is None or step <= 1:
        return arr
    return arr[::step]

def build_panel(rows: List[Dict[str, Any]], *, downsample_step: int) -> List[np.ndarray]:
    return [downsample_td(ts_dict_to_np(r["ts"]), downsample_step) for r in rows]

def pad_stack(X_list: List[np.ndarray]) -> np.ndarray:
    """Pad along time to common length; stack -> (N, T, D)."""
    T = max(x.shape[0] for x in X_list)
    D = X_list[0].shape[1]
    out = np.empty((len(X_list), T, D), dtype="float32")
    for i, x in enumerate(X_list):
        t = x.shape[0]
        if t < T:
            out[i] = np.pad(x, ((0, T - t), (0, 0)), mode="edge")
        else:
            out[i] = x
    return out

def make_class_lists(options: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Returns (letters_in_order, classnames_in_same_order)."""
    letters = sorted(options.keys())  # ['A','B',...]
    classes = [options[a] for a in letters]
    return letters, classes

def human_to_letter(pred_str: str, letters: List[str], classes: List[str]) -> str:
    try:
        idx = classes.index(pred_str)
        return letters[idx]
    except ValueError:
        return "?"


# -------------------- args & setup --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--input_path", required=True)
    p.add_argument("--train_path", required=False, default=None,
                   help="If omitted, will try to derive by replacing '/test.jsonl' with '/train.jsonl'")
    p.add_argument("--output_path", required=True)

    # compatibility flags (accepted but not used for modeling)
    p.add_argument("--shots", default="zs")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--debug_prints", type=int, default=0)

    # kNN params
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--metric", type=str, default="dtw", choices=["dtw", "euclidean"])
    p.add_argument("--sakoe_radius", type=int, default=10,
                   help="Sakoe–Chiba radius (only used for metric=dtw)")
    p.add_argument("--downsample", type=int, default=2,
                   help="Stride for downsampling along time (>=1; 1 = no downsampling)")
    p.add_argument("--n_jobs", type=int, default=-1)
    return p.parse_args()


def set_up(args: argparse.Namespace) -> Tuple[List[dict], List[dict], MasterLogger]:
    logger = MasterLogger(log_path="./logs/knn_classifier.log", init=True, clear=False)
    logger.info(f"Arguments: {args}")

    # derive train path if missing
    if args.train_path is None:
        if args.input_path.endswith("test.jsonl"):
            args.train_path = args.input_path.replace("/test.jsonl", "/train.jsonl")
        else:
            logger.error("train_path not provided and could not be derived.")
            sys.exit(1)

    test_data = load_jsonl(args.input_path)
    train_data = load_jsonl(args.train_path)

    if args.debug:
        train_data = train_data[:400]  # keep small for quick sanity
        test_data = test_data[:40]

    # prepare/clear output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write("")
    logger.info(f"Loaded train={len(train_data)} test={len(test_data)}")
    logger.info(f"Output -> {args.output_path}")
    return train_data, test_data, logger


# -------------------- main --------------------

def main():
    args = parse_args()
    train, test, logger = set_up(args)

    # Build panels
    X_train_list = build_panel(train, downsample_step=args.downsample)
    X_test_list  = build_panel(test,  downsample_step=args.downsample)

    # Pad + stack to (N, T, D)
    X_train = pad_stack(X_train_list)
    X_test  = pad_stack(X_test_list)

    # Labels (human-readable class strings)
    y_train = np.array([str(r["answer"]) for r in train], dtype=object)
    y_test  = np.array([str(r["answer"]) for r in test], dtype=object)

    # z-normalize per sample & dimension
    scaler = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Configure classifier
    knn_kwargs = dict(n_neighbors=args.k, n_jobs=args.n_jobs)
    if args.metric == "dtw":
        knn_kwargs.update({
            "metric": "dtw",
            "metric_params": {"global_constraint": "sakoe_chiba",
                              "sakoe_chiba_radius": args.sakoe_radius}
        })
    else:
        knn_kwargs.update({"metric": "euclidean"})

    clf = KNeighborsTimeSeriesClassifier(**knn_kwargs)
    clf.fit(X_train, y_train)

    # For letter mapping per-row we need the option map:
    # assume all rows share same option ordering; fall back per-row if needed
    first_opts = test[0]["options"]
    letters_order, classes_order = make_class_lists(first_opts)

    # Predict with tqdm over test set
    print(f"Predicting with k={args.k}, metric={args.metric}, downsample={args.downsample}, n_jobs={args.n_jobs}")
    for i, item in enumerate(tqdm(test, desc="Classifying")):
        # slice keeps shape (1, T, D)
        pred_str = clf.predict(X_test[i:i+1])[0]
        pred_letter = human_to_letter(pred_str, letters_order, classes_order)

        # Compose output row compatible with your writer
        out_row = {
            "idx": item.get("idx"),
            "question": item.get("question"),
            "PRED": {
                "final_answer": pred_letter,
                "pred_str": pred_str
            },
            "task_type": item.get("task_type", "classification"),
            "GT": item.get("answer"),
            "label": item.get("label"),
            "parent_task": "NA",
            "application_domain": "NA",
            "eval_type": "classification"
        }
        save_output(args, out_row)

    # Quick summary
    y_pred = np.array([clf.predict(X_test[i:i+1])[0] for i in range(len(test))], dtype=object)
    acc = (y_pred == y_test).mean() if len(y_test) else float("nan")
    logger.info(f"Final accuracy on this split: {acc:.3f}")


if __name__ == "__main__":
    main()
