# '''
# 2025-09-19
# Author: Dan Schumacher
# How to run:
#    see ./bin/vl_time.sh
# '''


# STANDARD IMPORTS
import argparse
import os
from typing import Tuple, List
from tqdm import tqdm

# USER DEFINED
from utils.argparsers import vl_time_parse_args
from utils.data_utils import save_output
from utils.logging_utils import MasterLogger
from utils.file_io import load_jsonl
from utils.vl_time_utils import work


def set_up() -> Tuple[List[dict], argparse.Namespace, MasterLogger]:
    # set up logger and args
    logger = MasterLogger(log_path="./logs/vl_time.log", init=True, clear=True)
    args = vl_time_parse_args()
    logger.info(f"Arguments: {args}")

    # load data
    data = load_jsonl(args.input_path)
    logger.info(f"Loaded {len(data)} examples from {args.input_path}")

    # prepare output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        f.write("")
    logger.info(f"Cleared output file: {args.output_path}")

    return data, args, logger


def main():
    # set up
    data, args, logger = set_up()
    dataset = args.input_path.split("/")[3]
    subset = args.input_path.split("/")[4]
    logger.info(f"Subset: {subset}")

    if args.debug:
        data = data[:3]
    # USED FOR FEW-SHOT EXAMPLES
    if args.shots == "zs":
        demo_pool = [] # no example, keep it empty
    elif args.shots == "fs":
        if subset=="ECG":
            print("FEW SHOT IS UNAVAILABLE FOR ECG. EXITING...")
            exit()
        demo_pool = load_jsonl(f"./data/examples/{dataset}/{subset}_shots.jsonl")
    else:
        raise ValueError("args.shots must be zs or fs")

    for item in tqdm(data, desc="Processing items"):
        logger.info(f"Processing item: {item}")

        classes = [str(c) for c in item["options"].values()]
        
        # ["CLASS1", "CLASS2", ...]
        class_desp = [f"[{chr(ord('A')+i)}] {c}" for i, c in enumerate(classes)]

        # ["A: CLASS1", "B: CLASS2", ...]
        class_desp = [f"{k}: {v}" for k, v in zip(item["options"].keys(), item["options"].values())]

        out_row = work(
            item,
            model="gpt-4o-mini-2024-07-18",
            modal="LV",                         # "V", "LV", or "L"
            seed=66,
            temperature=0.0,
            detail="low",
            image_token="<<IMG>>",
            classes=classes,
            class_desp=class_desp,
            demo_pool=demo_pool,
            num_shot_per_class=1,
            save_folder=f"./data/images/{dataset}/{subset}", # images created here on-the-fly
            file_suffix=".png",
            question=item["question"],
            prior_knowledge="",   
            hint="please think step by step",
            real_call=True,                     # set False for dry-run structure
            debug=args.debug_prints
        )
        save_output(args, out_row)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()