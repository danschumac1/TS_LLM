'''
2025-08-06
Author: Dan Schumacher
How to run:
   see ./bin/method/baseline.sh
'''

import sys
sys.path.append("./src")

import json
import os
from typing import Tuple, Union
import argparse
from tqdm import tqdm

from utils.baseline_utils import select_most_similar
from utils.data_utils import load_jsonl, save_output
from utils.get_prompter import get_prompter_class_and_kwargs
from utils.logging_utils import MasterLogger
from utils.prompter import GemmaPrompter, HFPrompter, OpenAIPrompter, QAs
from utils.argparsers import baseline_parse_args

def set_up() -> Tuple[argparse.Namespace, Union[GemmaPrompter, HFPrompter, OpenAIPrompter], MasterLogger]:
    args = baseline_parse_args()
    args.dataset = args.input_path.split("/")[3]
    args.subset = args.input_path.split("/")[4]
    logger = MasterLogger(log_path="./logs/baseline.log", init=True, clear=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger.info(f"Clearing output file: {args.output_path}")
    with open(args.output_path, 'w') as f:
        f.write("")

    show_prompt = bool(args.show_prompt)

    prompt_path = args.prompt_path
    prompt_headers = {
        "question": "Question:",
        "timeseries": "Time Series:",
    }

    # Get prompter class + fully prepared kwargs (now includes TimeMQA_* adapter settings)
    PrompterClass, shared_kwargs = get_prompter_class_and_kwargs(
        model_type=args.model_type,
        device_map=args.device_map,
        show_prompt=show_prompt,
    )

    prompter = PrompterClass(
        prompt_path=prompt_path,
        prompt_headers=prompt_headers,
        temperature=args.temperature,
        **shared_kwargs
    )

    if args.n_shots <= 0:
        demo_pool = [] # no example, keep it empty
    else:
        if args.subset=="ECG":
            print("FEW SHOT IS UNAVAILABLE FOR ECG. EXITING...")
            exit()
        elif "_TINY_TEST" in args.input_path:
            demo_pool = []
            for subset in ["CTU", "ECG", "EMG", "HAR", "TEE"]:
                sub_demo_pool = load_jsonl(f"./data/datasets/{args.dataset}/{subset}/train.jsonl")
                for line in sub_demo_pool:
                    line["subset"] = subset
                demo_pool.extend(sub_demo_pool)
        demo_pool = load_jsonl(f"./data/datasets/{args.dataset}/{args.subset}/train.jsonl")

    return args, prompter, logger, demo_pool

def main():
    args, prompter, logger, demo_pool = set_up()

    data = load_jsonl(args.input_path)
    if "_TINY_TEST" in args.input_path and args.n_shots>0:
        # filter out ECG
        print("No training data for ECG. Discluding from tiny test few-shot data.")
        data = [row for row in data if row['subset']!="ECG"]

    logger.info(f"Loaded {len(data)} input records from {args.input_path}")

    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches"):
        batch = data[i:i + args.batch_size]
        logger.info(f"Processing batch {i // args.batch_size + 1} (size={len(batch)})")

        # 1) pick top-k demos per row
        per_row_shots = [select_most_similar(row, demo_pool, n_examples=args.n_shots)
                         for row in batch]

        # 2) wrap each row’s demos as QAs (don’t mutate prompter.examples)


        per_input_examples = [
            [QAs(
                question={
                    "question": ex["question"],
                    "timeseries": json.dumps(ex["ts"])},  # you can add more fields here
                answer=f"The answer is: {ex["label"]} | {ex["answer"]}")
             for ex in shots]
            for shots in per_row_shots
        ]
        # 3) build inputs for this batch
        question_batch = [
            {
                "question": row["question"],                 # text question
                "timeseries": json.dumps(row["ts"], indent=2)  # extra field if your prompt uses it
            }
            for row in batch
        ]

        # 4) call once, passing per-input few-shots
        results = prompter.get_completion(
            question_batch,
            parse=True,
            per_input_examples=per_input_examples,   # ← key line
        )

        if not isinstance(results, list):
            results = [results]
            
        for input_dict, output in zip(batch, results):
            dict_line = {
                "idx": input_dict.get("idx", "ERROR"),
                "question": input_dict.get("question", "ERROR"),
                "PRED": output,
                "task_type": input_dict.get("task_type", "ERROR"),
                "GT": input_dict.get("answer", "ERROR"),
                "label": input_dict.get("label", "ERROR"),
                "parent_task": input_dict.get("parent_task", "NA"),
                "application_domain": input_dict.get("application_domain", "NA"),
                "eval_type": input_dict.get("eval_type", "classification"),
            }
            save_output(args, dict_line)

    logger.info(f"All results saved to {args.output_path}")

if __name__ == "__main__":
    main()
