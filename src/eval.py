"""
2025-09-09
Author: Dan Schumacher

Usage:
    python ./src/eval.py --input_path ./data/generations/gpt/MCQ1/baseline/few_shot/test.jsonl
"""

import argparse
import os
from datetime import datetime
from utils.data_utils import load_jsonl

def parse_args():
    p = argparse.ArgumentParser(description="Minimal MCQ eval → TSV")
    p.add_argument("--input_path", type=str, required=True)
    return p.parse_args()

def get_pred_label(pred):
    # Few-shot dict: {"final_answer": "A", ...}
    if isinstance(pred, dict):
        return pred.get("final_answer")
    # Zero-shot string formats we use: "The answer is: A" or "FINAL: A"
    if isinstance(pred, str):
        s = pred.strip()
        if "The answer is:" in s:
            return s.split("The answer is:")[-1].strip()
        if "FINAL:" in s:
            return s.split("FINAL:")[-1].strip()
        return s.strip()
    return None

def main():
    args = parse_args()

    # Expect: ./data/generations/<model>/<dataset>/<prompt>/<shot>/<split>.jsonl
    parts = args.input_path.replace("\\", "/").split("/")
    model   = parts[-5] if len(parts) >= 5 else "unknown_model"
    dataset = parts[-4] if len(parts) >= 4 else "unknown_dataset"
    prompt  = parts[-3] if len(parts) >= 3 else "unknown_prompt"
    shot    = parts[-2] if len(parts) >= 2 else "unknown_shot"
    split   = os.path.splitext(parts[-1])[0] if parts else "unknown_split"

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

    # Write/update final TSV
    results_path = "./data/results/results.tsv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    header = ["dataset", "split", "model", "prompt", "shot", "eval_date", "acc"]
    row = [
        dataset,
        split,
        model,
        prompt,
        shot,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{acc:.4f}",
    ]

    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
    with open(results_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(row) + "\n")

    # Also echo summary to stdout
    print(f"{dataset}\t{split}\t{model}\t{prompt}\t{shot}\t{row[-2]}\t{row[-1]}")

if __name__ == "__main__":
    main()
