"""
Created on 04/25/2025
@author: Dan Schumacher
HOW TO RUN: (see ./bin/TSQA_eval.sh)
"""

# LOCAL IMPORTS
from datetime import datetime
import json
import os
from typing import List, Tuple
from tqdm import tqdm
import argparse
from data_managment.TSQA.eval_type import EvalType, detect_eval_type
from utils.data_utils import load_jsonl
from utils.eval_utils import (
    eval_tf,
    eval_mc,
    calc_classification_acc,
    calc_anomaly_acc,
    calc_MSE,
)
from utils.logging_utils import MasterLogger, StandAloneLogger
from utils.prompter import OpenAIPrompter

def resolve_eval_type(line: dict, logger: MasterLogger) -> EvalType:
    raw = line.get("eval_type")
    if raw:
        try:
            return EvalType(raw)
        except ValueError:
            logger.warning(f"Unknown eval_type string in data: {raw}; falling back to heuristic.")
    logger.warning("No eval_type found in line; attempting to detect from content.")
    return detect_eval_type(line)  # fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate generated answers.')
    # REQUIRED arguments
    parser.add_argument(
        '--input_path', type=str, help='Path to the generated file', required=True)
    # OPTIONAL arguments
    return parser.parse_args()

from pathlib import Path

def set_up() -> Tuple[argparse.Namespace, MasterLogger, List[dict]]:
    args = parse_args()
    logger = MasterLogger(log_path="./logs/eval.log", init=True, clear=True)

    os.makedirs("./data/results", exist_ok=True)
    print(args.input_path)

    p = Path(args.input_path)
    # Expect: .../generations/{model}/{task}/{dataset}/{split}.jsonl
    try:
        parts = p.parts
        gi = parts.index("generations")
        model  = parts[gi+1]
        # task = parts[gi+2]  # unused here, but available
        dataset = parts[gi+3]
    except Exception:
        logger.warning("Could not parse model/dataset from input_path; defaulting to 'unknown'.")
        model = "unknown"
        dataset = "unknown"
    args.model = model
    args.dataset = dataset

    logger.info(f"Model: {model} | Dataset: {dataset}")
    logger.info(f"Input Path: {args.input_path}")

    args.results_path = "./data/results/results.tsv"
    data = load_jsonl(args.input_path)

    return args, logger, data



def main():
    args, logger, data = set_up()
    print(len(data), "predictions loaded")

    eval_function_map = {
        EvalType.TRUE_FALSE: eval_tf,
        EvalType.MULTIPLE_CHOICE: eval_mc,
        EvalType.FORECASTING_MSE: calc_MSE,
        EvalType.IMPUTATION_MSE: calc_MSE,
        EvalType.ANOMALY: calc_anomaly_acc,
        EvalType.CLASSIFICATION: calc_classification_acc,
        # OTHER_OPEN_ENDED_QA is intentionally unscored
    }


    results = {et: [] for et in EvalType}
    skipped_open_ended = 0
    skipped_none = 0
    skipped_unknown = 0

    for line in tqdm(data, desc="Evaluating answers"):
        et = resolve_eval_type(line, logger)

        if et == EvalType.OTHER_OPEN_ENDED_QA:
            skipped_open_ended += 1
            continue

        func = eval_function_map.get(et)
        if func is None:
            logger.warning(f"No eval function for eval_type={et.value}; skipping.")
            skipped_unknown += 1
            continue

        score = func(line)
        if score is None:
            logger.warning("Skipping line due to None score")
            skipped_none += 1
            continue

        results[et].append(score)


    print("\n=== Evaluation Results ===")
    for et, scores in results.items():
        if et == EvalType.OTHER_OPEN_ENDED_QA:
            continue
        if not scores:
            print(f"{et.value}: No samples.")
            continue

        if et in (EvalType.FORECASTING_MSE, EvalType.IMPUTATION_MSE):
            avg = sum(scores) / len(scores)
            print(f"{et.value.upper()}: {avg:.4f} (lower is better)  [n={len(scores)}]")
        else:
            acc = sum(scores) / len(scores)
            print(f"{et.value.upper()}: {acc*100:.2f}%  [n={len(scores)}]")

    print("\nSkipped counts:",
          f"open_ended={skipped_open_ended}, none_score={skipped_none}, unknown_type={skipped_unknown}")


    # Save raw results (exclude open-ended)
    save_path = args.results_path
    with open(save_path, "w") as f:
        f.write("eval_type\tscore\n")
        for et, scores in results.items():
            if et == EvalType.OTHER_OPEN_ENDED_QA:
                continue
            for s in scores:
                f.write(f"{et.value}\t{s}\n")
    print(f"\nSaved raw results to {save_path}")

if __name__ == "__main__":
    main()