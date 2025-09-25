"""
2025-09-21
Author: Dan Schumacher
Purpose: Cross-validate kNN (DTW/Euclidean) on TRAIN and append best params to a TSV.

Output TSV columns (tab-separated):
date_utc, dataset, subset, train_path, downsample_step, T_eff,
k, metric, sakoe_radius, cv_score, cv_metric, cv_folds, seed, n_jobs

How to run (single):
  python ./src/method/knn_cv.py \
    --train_path ./data/datasets/TimerBed/EMG/train.jsonl \
    --params_tsv ./data/experiments/knn/knn_cv_params.tsv \
    --cv_folds 5 --scoring accuracy --seed 66 --n_jobs -1 --debug 0
"""

import os, sys, argparse, csv, datetime
sys.path.append("./src")
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from utils.file_io import load_jsonl
from utils.logging_utils import MasterLogger
from utils.knn_helpers import build_panel, pad_stack

# -------------------- small helpers --------------------

def _parse_ds_subset_from_path(train_path: str) -> Tuple[str, str]:
    # expects .../data/datasets/{dataset}/{subset}/train.jsonl
    parts = os.path.normpath(train_path).split(os.sep)
    try:
        i = parts.index("datasets")
        dataset, subset = parts[i+1], parts[i+2]
        return dataset, subset
    except Exception:
        return "UNKNOWN", "UNKNOWN"

def _pick_downsample(train_rows: List[dict]) -> int:
    ts = train_rows[0]["ts"]
    if isinstance(ts, dict):
        any_dim = next(iter(ts.keys()))
        T = len(ts[any_dim])
    else:
        T = len(ts)
    if T >= 1000: return 3
    if T >= 512:  return 2
    return 1

def _dtw_radius_grid(T_eff: int) -> List[int]:
    pct = [0.05, 0.10, 0.15]
    return sorted({max(1, int(round(p * T_eff))) for p in pct})

def _estimate_effective_length(series_list: List[np.ndarray]) -> int:
    lengths = [s.shape[0] for s in series_list if s is not None]
    return int(np.median(lengths)) if lengths else 1

def _scorer(name: str):
    return make_scorer(f1_score, average="macro") if name == "macro_f1" else make_scorer(accuracy_score)

def _candidate_grid(T_eff: int) -> List[Dict[str, Any]]:
    grid = []
    for k in (1, 3):
        grid.append({"metric": "euclidean", "k": k})
        for r in _dtw_radius_grid(T_eff):
            grid.append({"metric": "dtw", "k": k, "sakoe_radius": r})
    return grid

def _choose_cv_folds(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    feasible = int(np.clip(requested, 2, counts.min()))
    return max(2, feasible)

def _fit_and_score_cv(X, y, cfg, cv_folds, seed, scoring, n_jobs) -> float:
    if cfg["metric"] == "dtw":
        knn = KNeighborsTimeSeriesClassifier(
            n_neighbors=cfg["k"],
            metric="dtw",
            metric_params={"global_constraint": "sakoe_chiba",
                           "sakoe_chiba_radius": cfg["sakoe_radius"]},
            n_jobs=n_jobs,
        )
    else:
        knn = KNeighborsTimeSeriesClassifier(
            n_neighbors=cfg["k"],
            metric="euclidean",
            n_jobs=n_jobs,
        )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(knn, X, y, cv=skf, scoring=_scorer(scoring), n_jobs=1)
    return float(np.mean(scores))

# -------------------- args --------------------

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--params_tsv", type=str, required=True)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--scoring", type=str, default="accuracy", choices=["accuracy", "macro_f1"])
    ap.add_argument("--seed", type=int, default=66)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--debug", type=int, default=0)
    return ap.parse_args()

# -------------------- main --------------------

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.params_tsv), exist_ok=True)
    logger = MasterLogger(log_path="./logs/knn_cv.log", init=True, clear=False)
    logger.info(f"[CV] args = {args}")

    dataset, subset = _parse_ds_subset_from_path(args.train_path)
    train = load_jsonl(args.train_path)
    if args.debug:
        train = train[:min(128, len(train))]

    downsample_step = _pick_downsample(train)
    X_train_list = build_panel(train, downsample_step=downsample_step)
    T_eff = _estimate_effective_length(X_train_list)
    X_train = pad_stack(X_train_list)

    y_train = np.array([str(r["answer"]) for r in train], dtype=object)

    scaler = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)

    cv_folds = _choose_cv_folds(y_train, args.cv_folds)
    if cv_folds != args.cv_folds:
        logger.info(f"Adjusted cv_folds from {args.cv_folds} -> {cv_folds}")

    grid = _candidate_grid(T_eff)
    logger.info(f"Grid size: {len(grid)} candidates; scoring={args.scoring}")

    best, best_score = None, -np.inf
    for cfg in tqdm(grid, desc=f"CV {dataset}/{subset} ({cv_folds}-fold)"):
        score = _fit_and_score_cv(X_train, y_train, cfg, cv_folds, args.seed, args.scoring, args.n_jobs)
        # tie-break: higher score; then smaller k; then prefer dtw
        if (score > best_score) or (
            np.isclose(score, best_score) and (
                (best is None or cfg["k"] < best["k"]) or
                (cfg["metric"] == "dtw" and best and best["metric"] != "dtw")
            )
        ):
            best, best_score = cfg, score

    assert best is not None
    logger.info(f"BEST: {best} | CV-{args.scoring}={best_score:.4f} | ds={downsample_step} T_eff={T_eff}")

    # append to TSV with header if needed
    new_file = not os.path.exists(args.params_tsv)
    with open(args.params_tsv, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if new_file:
            writer.writerow(["date_utc","dataset","subset","train_path","downsample_step","T_eff",
                             "k","metric","sakoe_radius","cv_score","cv_metric","cv_folds","seed","n_jobs"])
        writer.writerow([
            datetime.datetime.utcnow().isoformat(timespec="seconds"),
            dataset, subset, args.train_path, downsample_step, T_eff,
            best["k"], best["metric"], best.get("sakoe_radius",""),
            round(float(best_score), 6), args.scoring, cv_folds, args.seed, args.n_jobs
        ])

if __name__ == "__main__":
    main()
