'''
2025-08-06
Author: Dan Schumacher
How to run:
   python ./src/data_managment/TSQA/load_TSQA.py
'''

from collections import Counter
import sys
sys.path.append("./src")
import os
import random
from tqdm import tqdm
from utils.data_utils import load_csv, save_jsonl, normalize_cols, parse_qa_list_TSQA, assign_idx_key
from data_managment.TSQA.eval_type import detect_eval_type, EvalType

def main():
    random.seed(42)
    counts = Counter()
    
    pre_path = "./data/datasets/raw_data/TSQA/"
    output_path = "./data/datasets/TSQA/"
    os.makedirs(output_path, exist_ok=True)

    tasks = ["anomaly_detection", "classification", "forecasting_imputation", "open_ended_qa"]
    
    full_paths = []
    for task in tasks:
        if task == "forecasting_imputation":
            for version in ["1", "2"]:
                full_paths.append(f"{pre_path}{task}/{task}{version}.csv")
        else:
            full_paths.append(f"{pre_path}{task}/{task}.csv")

    full_data = {}
    for path in full_paths:
        print(f"Loading {path}")
        file_name = os.path.splitext(os.path.basename(path))[0]  # e.g., forecasting_imputation1
        task_name = file_name  # use full filename as task key
        if task_name not in full_data:
            full_data[task_name] = []
        data = load_csv(path)
        full_data[task_name].extend(data)

    print(f"Loaded {len(full_data)} tasks with data")

    all_errors = 0
    full_clean_data = {}

    for task, data_list in full_data.items():
        print("-" * 50)
        print(f"Task: {task}, Number of items: {len(data_list)}")

        clean_data = []
        task_errors = 0
        for item in tqdm(data_list, desc=f"Processing {task}"):
            item = normalize_cols(item)
            parsed_item, task_errors = parse_qa_list_TSQA(item, task_errors)
            if parsed_item is not None:
                parsed_item = normalize_cols(parsed_item)
                parsed_item = {k: v for k, v in parsed_item.items() if k != "qa_list" and v is not None}
                parsed_item["parent_task"] = task
                
                eval_type = detect_eval_type(parsed_item)
                parsed_item["eval_type"] = eval_type.value
                counts[eval_type.value] += 1
                clean_data.append(parsed_item)

        all_errors += task_errors
        full_clean_data[task] = clean_data
        print(f"Task: {task}, Errors found: {task_errors}")
    print(f"Total errors across all tasks: {all_errors}")
    print("\n=== Eval Type Counts ===")
    for etype, count in counts.items():
        print(f"{etype}: {count}")


        # Save per-task output
        # save_jsonl(clean_data, os.path.join(output_path, f"{task}_clean.jsonl"))

    # SEPERATE IMPUTATION AND FORECASTING DATA AND STITCH THEM TOGETHER
    imputation1 = [line for line in full_clean_data["forecasting_imputation1"] if line.get("task_type") == "imputation"]
    imputation2 = [line for line in full_clean_data["forecasting_imputation2"] if line.get("task_type") == "imputation"]
    forecasting1 = [line for line in full_clean_data["forecasting_imputation1"] if line.get("task_type") == "forecasting"]
    forecasting2 = [line for line in full_clean_data["forecasting_imputation2"] if line.get("task_type") == "forecasting"]
    full_clean_data["imputation"] = imputation1 + imputation2
    full_clean_data["forecasting"] = forecasting1 + forecasting2
    full_clean_data = {k: v for k, v in full_clean_data.items() if k not in ["forecasting_imputation1", "forecasting_imputation2"]}

    for k, v in full_clean_data.items():
        v = assign_idx_key(v, idx_key="idx")
        full_clean_data[k] = v

    # === Optional: save sampled train/dev/test split ===
    full_train_data = []
    full_dev_data = []
    full_test_data = []
    for task, data_list in full_clean_data.items():
        print(f"Task: {task}, Number of items: {len(data_list)}")
        random.shuffle(data_list)  # Shuffle the data for randomness
        n_total = len(data_list)
        n_train = min(200, n_total)
        n_dev = min(40, n_total - n_train)
        n_test = min(40, n_total - n_train - n_dev)
        train = data_list[:n_train]
        dev = data_list[n_train:n_train + n_dev]
        test = data_list[n_train + n_dev:n_train + n_dev + n_test]
        full_train_data.extend(train)
        full_dev_data.extend(dev)
        full_test_data.extend(test)
    random.shuffle(full_train_data)  # Shuffle the final splits for randomness
    for lst in [full_train_data, full_dev_data, full_test_data]:
        random.shuffle(lst)  # Shuffle each split for randomness
    print(f"Total train items: {len(full_train_data)}")
    save_jsonl(full_train_data, os.path.join(output_path, "train.jsonl"))
    print(f"Total dev items: {len(full_dev_data)}")
    save_jsonl(full_dev_data, os.path.join(output_path, "dev.jsonl"))
    print(f"Total test items: {len(full_test_data)}")
    save_jsonl(full_test_data, os.path.join(output_path, "test.jsonl"))
    flat_clean_data = [item for sublist in full_clean_data.values() for item in sublist]
    print(f"Total cleaned data items: {len(flat_clean_data)}")
    save_jsonl(flat_clean_data, os.path.join(output_path, "full_clean_data.jsonl"))

    # random.shuffle(full_clean_data)

    # n_total = len(full_clean_data)
    # n_train = min(1000, n_total)
    # n_dev = min(200, n_total - n_train)
    # n_test = min(200, n_total - n_train - n_dev)

    # train = full_clean_data[:n_train]
    # dev = full_clean_data[n_train:n_train + n_dev]
    # test = full_clean_data[n_train + n_dev:n_train + n_dev + n_test]

    # sampled_splits = {
    #     'train': train,
    #     'dev': dev,
    #     'test': test
    # }

    # for split, data_list in sampled_splits.items():
    #     print(f"Saving sampled {split} split with {len(data_list)} items")
    #     save_jsonl(data_list, os.path.join(output_path, f"{split}_sample.jsonl"))


if __name__ == "__main__":
    main()
