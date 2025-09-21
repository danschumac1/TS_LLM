"""
2025-09-20
Author: Dan Schumacher (updated)

Usage:
    python ./src/eval.py --input_path <path-to-a-generated-jsonl>

Supports two layouts:

LLM-style (method uses shots/model):
  ./data/generations/<method>/<model>/<dataset>/<subset>/<shot>/<split>.jsonl
  e.g. ./data/generations/vl_time/gpt/TimerBed/CTU/zs/test.jsonl

kNN-style (no model/shots; hyperparams encoded in folder name):
  ./data/generations/knn_classifier/<dataset>/<subset>/<param_slug>/<split>.jsonl
  e.g. ./data/generations/knn_classifier/TimerBed/CTU/k=1_metric=dtw_band=10_ds=2/test.jsonl
"""

import argparse
import os
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any

# keep your original import location
from utils.data_utils import load_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Minimal MCQ eval → TSV (LLM + kNN)")
    p.add_argument("--input_path", type=str, required=True)
    return p.parse_args()


def get_pred_label(pred: Any) -> Optional[str]:
    # Few-shot dict: {"final_answer": "A", ...}
    if isinstance(pred, dict):
        return pred.get("final_answer")
    # Zero-shot string formats: "The answer is: A" or "FINAL: A"
    if isinstance(pred, str):
        s = pred.strip()
        if "The answer is:" in s:
            return s.split("The answer is:")[-1].strip()
        if "FINAL:" in s:
            return s.split("FINAL:")[-1].strip()
        return s.strip()
    return None


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_path(path: str) -> Dict[str, str]:
    """
    Returns a dict with keys:
      dataset, subset, split, method, model, prompt, shot, params

    - For kNN: method='knn_classifier', model='knn', shot='na', params='<param_slug>'
    - For LLM: params='NA'
    """
    parts = path.replace("\\", "/").split("/")
    split = _stem(path)

    # try to find anchor "generations"
    try:
        gidx = parts.index("generations")
    except ValueError:
        gidx = None

    out = {
        "dataset": "unknown_dataset",
        "subset": "unknown_subset",
        "split": split,
        "method": "unknown_method",
        "model": "unknown_model",
        "prompt": "unknown_prompt",
        "shot": "unknown_shot",
        "params": "NA",
    }

    if gidx is not None and gidx + 1 < len(parts):
        # kNN layout?
        if "knn_classifier" in parts:
            # .../generations/knn_classifier/<dataset>/<subset>/<param_slug>/<split>.jsonl
            try:
                kidx = parts.index("knn_classifier")
                out["method"] = "knn_classifier"
                out["model"] = "knn"
                out["dataset"] = parts[kidx + 1]
                out["subset"] = parts[kidx + 2]
                out["params"] = parts[kidx + 3]  # e.g., k=1_metric=dtw_band=10_ds=2
                out["shot"] = "na"
                out["split"] = split
                return out
            except Exception:
                pass

        # LLM layout (method first):
        # .../generations/<method>/<model>/<dataset>/<subset>/<shot>/<split>.jsonl
        try:
            method = parts[gidx + 1]
            model = parts[gidx + 2]
            dataset = parts[gidx + 3]
            subset = parts[gidx + 4]
            shot = parts[gidx + 5]
            out.update({
                "method": method,
                "model": model,
                "dataset": dataset,
                "subset": subset,
                "shot": shot,
                "prompt": method,     # keep legacy column semantics
                "params": "NA",
                "split": split,
            })
            return out
        except Exception:
            # Fallback to older layout (model/dataset/prompt/shot)
            # .../generations/<model>/<dataset>/<prompt>/<shot>/<split>.jsonl
            try:
                model = parts[-5]
                dataset = parts[-4]
                prompt = parts[-3]
                shot = parts[-2]
                out.update({
                    "method": "na",
                    "model": model,
                    "dataset": dataset,
                    "subset": "na",
                    "prompt": prompt,
                    "shot": shot,
                    "params": "NA",
                    "split": split,
                })
                return out
            except Exception:
                return out

    return out


def main():
    args = parse_args()
    meta = parse_path(args.input_path)

    rows = load_jsonl(args.input_path)

    total, correct = 0, 0
    for row in rows:
        total += 1
        pred = get_pred_label(row.get("PRED"))
        gold = row.get("label")
        pred_norm = (pred or "").strip().upper()
        gold_norm = (str(gold) if gold is not None else "").strip().upper()
        if pred_norm and pred_norm == gold_norm:
            correct += 1

    acc = (correct / total) if total else 0.0

    # Append to TSV (now with subset + params)
    results_path = "./data/results/results.tsv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    header = ["dataset", "subset", "split", "model", "prompt", "shot", "params", "eval_date", "acc"]
    row = [
        meta["dataset"],
        meta["subset"],
        meta["split"],
        meta["model"],
        meta["prompt"],
        meta["shot"],
        meta["params"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{acc:.4f}",
    ]

    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
    with open(results_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(row) + "\n")

    # Console summary — includes HYPERPARAMETERS line if present
    print("------------------------------------------------------------")
    print(f"DATASET       : {meta['dataset']}")
    print(f"SUBSET        : {meta['subset']}")
    print(f"SPLIT         : {meta['split']}")
    print(f"METHOD        : {meta['method']}")
    print(f"MODEL         : {meta['model']}")
    print(f"SHOT          : {meta['shot']}")
    if meta["params"] != "NA":
        print(f"HYPERPARAMETERS: {meta['params']}")
    print(f"ACCURACY      : {acc:.4f}")
    print("------------------------------------------------------------")
    # Also echo one-liner (keeps your old behavior easy to parse)
    print(f"{meta['dataset']}\t{meta['subset']}\t{meta['split']}\t{meta['model']}\t{meta['prompt']}\t{meta['shot']}\t{meta['params']}\t{row[-2]}\t{row[-1]}")


if __name__ == "__main__":
    main()
