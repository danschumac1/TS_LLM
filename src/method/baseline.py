'''
2025-08-06
Author: Dan Schumacher
How to run:
   see ./bin/baseline.sh
'''
import json
import sys
sys.path.append("./src")
import os
from typing import Tuple, Union, List, Dict
import argparse
from tqdm import tqdm

from utils.data_utils import load_jsonl, save_output
from utils.get_prompter import get_prompter_class_and_kwargs
from utils.logging_utils import MasterLogger
from utils.prompter import GemmaPrompter, HFPrompter, OpenAIPrompter
from utils.argparsers import baseline_parse_args

def set_up() -> Tuple[argparse.Namespace, Union[GemmaPrompter, HFPrompter, OpenAIPrompter], MasterLogger]:
    args = baseline_parse_args()
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

    logger.info(f"Using {PrompterClass.__name__} with parameters: {shared_kwargs}")
    return args, prompter, logger

def main():
    args, prompter, logger = set_up()

    data = load_jsonl(args.input_path)
    # data = data[:10] # TODO remove limit
    logger.info(f"Loaded {len(data)} input records from {args.input_path}")

    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches"):
        logger.info(f"Processing batch {i // args.batch_size + 1} of {len(data[i:i + args.batch_size])}")
        batch = data[i:i + args.batch_size]
        question_batch = [
            {
                "question": example['question'], 
                'timeseries':json.dumps(example['ts'], indent=4)
            } for example in batch]

        results = prompter.get_completion(question_batch)
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
