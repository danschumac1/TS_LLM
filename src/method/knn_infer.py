"""
2025-09-21
Author: Dan Schumacher
Purpose: Train final kNN on full TRAIN using params from TSV and predict TEST -> JSONL.

How to run (single):
  python ./src/method/knn_infer.py \
    --input_path ./data/datasets/TimerBed/EMG/test.jsonl \
    --train_path ./data/datasets/TimerBed/EMG/train.jsonl \
    --params_tsv ./data/experiments/params/knn_cv_params.tsv \
    --output_path ./data/generations/knn_classifier/TimerBed/EMG/zs/test.jsonl \
    --select best  --cv_metric accuracy  --debug 0
"""

import os, sys, argparse, csv
sys.path.append("./src")
from typing import Tuple, Dict, Any, List
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils.file_io import load_jsonl
from utils.data_utils import save_output
from utils.logging_utils import MasterLogger
from utils.knn_helpers import build_panel, pad_stack, make_class_lists, human_to_letter

# -------------------- helpers --------------------

def _parse_ds_subset_from_path(path: str) -> Tuple[str, str]:
    parts = os.path.normpath(path).split(os.sep)
    try:
        i = parts.index("datasets")
        dataset, subset = parts[i+1], parts[i+2]
        return dataset, subset
    except Exception:
        return "UNKNOWN", "UNKNOWN"

def _load_best_row(params_tsv: str, dataset: str, subset: str, prefer_metric: str = "accuracy") -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(params_tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["dataset"] == dataset and r["subset"] == subset:
                rows.append(r)
    if not rows:
        raise RuntimeError(f"No TSV row for {dataset}/{subset} in {params_tsv}")

    # filter by metric if available
    metric_rows = [r for r in rows if r["cv_metric"] == prefer_metric] or rows
    # pick row with max cv_score
    metric_rows.sort(key=lambda r: float(r["cv_score"]), reverse=True)
    return metric_rows[0]

# -------------------- args --------------------

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--params_tsv", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--select", type=str, default="best", choices=["best"])  # future-friendly
    ap.add_argument("--cv_metric", type=str, default="accuracy", choices=["accuracy","macro_f1"])
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--debug", type=int, default=0)
    return ap.parse_args()

# -------------------- main --------------------

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger = MasterLogger(log_path="./logs/knn_infer.log", init=True, clear=False)
    logger.info(f"[INFER] args = {args}")

    dataset, subset = _parse_ds_subset_from_path(args.train_path)
    best = _load_best_row(args.params_tsv, dataset, subset, prefer_metric=args.cv_metric)

    # Parse params
    k = int(best["k"])
    metric = best["metric"]
    sakoe_radius = int(best["sakoe_radius"]) if str(best.get("sakoe_radius","")).strip() != "" else None
    downsample_step = int(best["downsample_step"])
    cv_folds = int(best["cv_folds"])
    cv_score = float(best["cv_score"])

    # Load data
    test = load_jsonl(args.input_path)
    train = load_jsonl(args.train_path)
    if args.debug:
        test = test[:min(8, len(test))]

    # Build panels with *the same* downsample_step used in CV
    X_train_list = build_panel(train, downsample_step=downsample_step)
    X_test_list  = build_panel(test,  downsample_step=downsample_step)
    X_train = pad_stack(X_train_list)
    X_test  = pad_stack(X_test_list)

    y_train = np.array([str(r["answer"]) for r in train], dtype=object)
    y_test  = np.array([str(r["answer"]) for r in test], dtype=object)

    scaler = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if metric == "dtw":
        knn_kwargs = {
            "n_neighbors": k,
            "metric": "dtw",
            "metric_params": {"global_constraint": "sakoe_chiba",
                              "sakoe_chiba_radius": sakoe_radius},
            "n_jobs": args.n_jobs,
        }
        band_slug = f"_band={sakoe_radius}"
    else:
        knn_kwargs = {"n_neighbors": k, "metric": "euclidean", "n_jobs": args.n_jobs}
        band_slug = ""

    clf = KNeighborsTimeSeriesClassifier(**knn_kwargs)
    clf.fit(X_train, y_train)

    # letter mapping
    first_opts = test[0]["options"] if test and "options" in test[0] else None
    if first_opts:
        letters_order, classes_order = make_class_lists(first_opts)
    else:
        classes_order = sorted(list({str(r["answer"]) for r in test}))
        letters_order = [chr(ord("A")+i) for i in range(len(classes_order))]

    param_slug = f"k={k}_metric={metric}{band_slug}_ds={downsample_step}_cv={cv_folds}fold"
    print(f"Predicting with {param_slug} | cv_{args.cv_metric}={cv_score:.4f}")

    # write JSONL lines compatible with your framework
    for i, item in enumerate(test):
        pred_str = clf.predict(X_test[i:i+1])[0]
        pred_letter = human_to_letter(pred_str, letters_order, classes_order)
        out_row = {
            "idx": item.get("idx"),
            "question": item.get("question"),
            "PRED": {
                "final_answer": pred_letter,
                "pred_str": pred_str,
                "params": param_slug,
                "cv_score": round(cv_score, 4),
                "cv_metric": args.cv_metric,
            },
            "task_type": item.get("task_type", "classification"),
            "GT": item.get("answer"),
            "label": item.get("label"),
            "parent_task": "NA",
            "application_domain": "NA",
            "eval_type": "classification",
        }
        save_output(args, out_row)

if __name__ == "__main__":
    main()
