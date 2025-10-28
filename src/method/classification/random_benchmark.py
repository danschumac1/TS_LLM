'''
2025-09-29
Author: Dan Schumacher
How to run:
   python ./src/method/random_benchmark.py
'''
import sys; sys.path.append("./src")
import argparse
import os
from typing import List, Tuple
import random

from tqdm import tqdm
from utils.argparsers import random_baseline_parser
from utils.data_utils import save_output
from utils.logging_utils import MasterLogger
from utils.file_io import load_jsonl

def select_random_choice(row: dict) -> Tuple[str, str, str]:
    options_list = list(row['options'].items())
    letter, answer = random.choice(options_list)
    output = f"The answer is: {letter} | {answer}"
    return output, letter, answer

# TODO
def stratified_random_choice(row:dict) -> Tuple[str, str, str]:
    # based on the PROPORTION OF ANSWER CHOICES IN TRAINING DATA
    options_list = list(row['options'].items())
    letter, answer = ...
    output = f"The answer is: {letter} | {answer}"
    return output, letter, answer

def set_up() -> Tuple[argparse.Namespace, MasterLogger, List[dict]]:
    args = random_baseline_parser()
    args.dataset = args.input_path.split("/")[3]
    args.subset = args.input_path.split("/")[4]
    logger = MasterLogger(log_path="./logs/baseline.log", init=True, clear=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger.info(f"Clearing output file: {args.output_path}")
    with open(args.output_path, 'w') as f:
        f.write("")

    data = load_jsonl(args.input_path)

    return args, logger, data

def main():
    random.seed(42)

    args, logger, data = set_up()

    logger.info(f"Loaded {len(data)} input records from {args.input_path}")

    for row in tqdm(data, total=len(data), desc="Random guessing"):

        output_str, letter, answer_text  = select_random_choice(row)
        pred_payload = {
            "final_answer": letter,
            "parsed_desc": answer_text,
            "raw_text": output_str,
        }

        dict_line = {
            "idx": row.get("idx", "ERROR"),
            "question": row.get("question", "ERROR"),
            "PRED": pred_payload,  # keep shape consistent with other methods
            "task_type": row.get("task_type", "ERROR"),
            "GT": row.get("answer", "ERROR"),
            "label": row.get("label", "ERROR"),
            "parent_task": row.get("parent_task", "NA"),
            "application_domain": row.get("application_domain", "NA"),
            "eval_type": row.get("eval_type", "classification"),
        }

        save_output(args, dict_line)

    logger.info(f"All results saved to {args.output_path}")


if __name__ == "__main__":
    main()