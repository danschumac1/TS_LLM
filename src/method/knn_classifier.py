"""
2025-09-20
Author: Dan Schumacher
Non-LLM classical baseline: kNN (DTW/Euclidean) for time series classification.

Changes:
- Performs cross-validation on the TRAIN split to select hyperparameters.
- Trains final model on full TRAIN with best params.
- Predicts TEST and writes JSONL lines compatible with your framework.
- Removes CLI hyperparameter flags (k/metric/sakoe_radius/downsample).

How to run:
    ./bin/method/knn_classifier.sh  (no hyperparameter args needed)
"""

import os, sys, argparse
sys.path.append("./src")
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

# Model & preprocessing
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# CV utilities
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Project utils
from utils.file_io import load_jsonl
from utils.data_utils import save_output
from utils.logging_utils import MasterLogger
from utils.knn_helpers import build_panel, pad_stack, make_class_lists, human_to_letter
from utils.argparsers import knn_parse_args  # updated below


# -------------------- helpers --------------------

def _pick_downsample(train_rows: List[dict]) -> int:
    """Heuristic: downsample long series; keep short untouched."""
    # infer raw length T from the first row (any dimension)
    ts = train_rows[0]["ts"]
    if isinstance(ts, dict):
        any_dim = next(iter(ts.keys()))
        T = len(ts[any_dim])
    else:
        T = len(ts)
    # simple heuristic
    if T >= 1000:
        return 3
    if T >= 512:
        return 2
    return 1


def _dtw_radius_grid(T_eff: int) -> List[int]:
    """Return a small, sane grid of Sakoe–Chiba radii in time points based on effective length."""
    # 5%, 10%, 15% of time length (rounded, unique, >=1)
    pct = [0.05, 0.10, 0.15]
    vals = sorted({max(1, int(round(p * T_eff))) for p in pct})
    return vals


def _estimate_effective_length(series_list: List[np.ndarray]) -> int:
    """Median length across series after downsampling (before padding)."""
    lengths = [s.shape[0] for s in series_list if s is not None]
    return int(np.median(lengths)) if lengths else 1


def _scorer(name: str):
    if name == "macro_f1":
        return make_scorer(f1_score, average="macro")
    return make_scorer(accuracy_score)


def _candidate_grid(T_eff: int) -> List[Dict[str, Any]]:
    """Small grid: metric in {euclidean, dtw}; k in {1,3}; dtw radius from length."""
    grid = []
    for k in (1, 3):
        # Euclidean
        grid.append({"metric": "euclidean", "k": k})
        # DTW with band
        for r in _dtw_radius_grid(T_eff):
            grid.append({"metric": "dtw", "k": k, "sakoe_radius": r})
    return grid


def _fit_and_score_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    cv_folds: int,
    seed: int,
    scoring: str,
    n_jobs: int,
) -> float:
    """Cross-validate a candidate configuration and return mean score."""
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

    # stratified CV (no groups available in current JSON)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        knn, X, y, cv=skf, scoring=_scorer(scoring), n_jobs=1  # n_jobs handled inside knn
    )
    return float(np.mean(scores))


def _choose_cv_folds(y: np.ndarray, requested: int) -> int:
    """Ensure folds are feasible given per-class counts."""
    # min class count across labels
    _, counts = np.unique(y, return_counts=True)
    feasible = int(np.clip(requested, 2, counts.min()))
    return max(2, feasible)


# -------------------- setup --------------------

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
        test_data = test_data[:min(8, len(test_data))]

    # prepare/clear output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write("")
    logger.info(f"Loaded train={len(train_data)} test={len(test_data)}")
    logger.info(f"Output -> {args.output_path}")
    return train_data, test_data, logger


# -------------------- main --------------------

def main():
    args = knn_parse_args()
    train, test, logger = set_up(args)

    # Decide downsample internally (no CLI flag)
    downsample_step = _pick_downsample(train)
    logger.info(f"Chosen downsample_step={downsample_step}")

    # Build panels (lists) then stack/pad
    X_train_list = build_panel(train, downsample_step=downsample_step)
    X_test_list  = build_panel(test,  downsample_step=downsample_step)

    # Effective length for DTW band grid
    T_eff = _estimate_effective_length(X_train_list)
    logger.info(f"Estimated effective time length (post-downsample) T_eff={T_eff}")

    # Stack to (N, T, D)
    X_train = pad_stack(X_train_list)
    X_test  = pad_stack(X_test_list)

    # Labels as human strings
    y_train = np.array([str(r["answer"]) for r in train], dtype=object)
    y_test  = np.array([str(r["answer"]) for r in test], dtype=object)

    # Per-sample z-normalization (safe; no leakage)
    scaler = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Choose feasible folds given class counts
    cv_folds = _choose_cv_folds(y_train, args.cv_folds)
    if cv_folds != args.cv_folds:
        logger.info(f"Adjusted cv_folds from {args.cv_folds} -> {cv_folds} (limited by class counts).")

    # Build small grid
    grid = _candidate_grid(T_eff)
    logger.info(f"Grid size: {len(grid)} candidates")

    # CV search
    best = None
    best_score = -np.inf
    for cfg in tqdm(grid, desc=f"CV ({cv_folds}-fold, scoring={args.scoring})"):
        score = _fit_and_score_cv(
            X_train, y_train,
            cfg, cv_folds=cv_folds, seed=args.seed,
            scoring=args.scoring, n_jobs=args.n_jobs
        )
        if args.debug_prints:
            logger.info(f"CFG {cfg} -> {args.scoring}={score:.4f}")
        # tie-breakers: prefer smaller k, then dtw over euclidean
        if (score > best_score) or (
            np.isclose(score, best_score) and (
                (cfg["k"] < best["k"]) if best else True or
                (cfg["metric"] == "dtw" and (best and best["metric"] != "dtw"))
            )
        ):
            best = cfg
            best_score = score

    assert best is not None, "No best configuration selected."
    logger.info(f"BEST CFG: {best}  | CV-{args.scoring}={best_score:.4f}")

    # Train final model on full train with BEST
    if best["metric"] == "dtw":
        knn_kwargs = {
            "n_neighbors": best["k"],
            "metric": "dtw",
            "metric_params": {"global_constraint": "sakoe_chiba",
                              "sakoe_chiba_radius": best["sakoe_radius"]},
            "n_jobs": args.n_jobs,
        }
        band_slug = f"_band={best['sakoe_radius']}"
    else:
        knn_kwargs = {"n_neighbors": best["k"], "metric": "euclidean", "n_jobs": args.n_jobs}
        band_slug = ""

    clf = KNeighborsTimeSeriesClassifier(**knn_kwargs)
    clf.fit(X_train, y_train)

    # Map human->letter for output
    first_opts = test[0]["options"] if test and "options" in test[0] else None
    if first_opts:
        letters_order, classes_order = make_class_lists(first_opts)
    else:
        classes_order = sorted(list({str(r["answer"]) for r in test}))
        letters_order = [chr(ord("A")+i) for i in range(len(classes_order))]

    # Predict & write JSONL, with a param tag into PRED for traceability
    param_slug = f"k={best['k']}_metric={best['metric']}{band_slug}_ds={downsample_step}_cv={cv_folds}fold"
    print(f"Predicting with {param_slug}")
    for i, item in enumerate(tqdm(test, desc="Classifying test")):
        pred_str = clf.predict(X_test[i:i+1])[0]
        pred_letter = human_to_letter(pred_str, letters_order, classes_order)
        out_row = {
            "idx": item.get("idx"),
            "question": item.get("question"),
            "PRED": {
                "final_answer": pred_letter,
                "pred_str": pred_str,
                "params": param_slug,
                "cv_score": round(float(best_score), 4),
                "cv_metric": args.scoring,
            },
            "task_type": item.get("task_type", "classification"),
            "GT": item.get("answer"),
            "label": item.get("label"),
            "parent_task": "NA",
            "application_domain": "NA",
            "eval_type": "classification",
        }
        save_output(args, out_row)

    # Quick test summary
    y_pred = np.array([clf.predict(X_test[i:i+1])[0] for i in range(len(test))], dtype=object)
    acc = float(np.mean(y_pred == y_test)) if len(y_test) else float("nan")
    logger.info(f"Final TEST accuracy: {acc:.4f}  | using {param_slug}")


if __name__ == "__main__":
    main()
