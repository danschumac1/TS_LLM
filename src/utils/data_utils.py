import re
import sys
from typing import Union
sys.path.append("./src")

import argparse
import json
import csv
import os
from json import loads as json_loads
from json.decoder import JSONDecodeError
from json_repair import repair_json
from utils.logging_utils import MasterLogger

#region BASIC UTILITIES
####################################################################################################
# BASIC UTILITIES
####################################################################################################
def load_json(file_path:str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data

def load_jsonl(file_path:str) -> dict:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    return data

def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"', escapechar='\\')
        return list(reader)

def save_json(data:dict, file_path:str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_jsonl(data:dict, file_path:str):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def fix_multiline_csv(input_path: str, output_path: str) -> None:
    """
    Rewrites the input CSV so that each row is fully on one line.
    It joins multi-line `qa_list` values and ensures each row ends cleanly.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        buffer = ""
        for line in infile:
            # Strip leading/trailing whitespace
            line = line.rstrip('\n')

            # Add current line to buffer
            if not buffer:
                buffer = line
            else:
                buffer += " " + line

            # Check if current buffer ends with a triple quote indicating record end
            if buffer.count('"""') % 2 == 0:  # naive: assumes qa_list is the only field with triple quotes
                outfile.write(buffer + '\n')
                buffer = ""

        if buffer:
            print("[Warning] Unclosed quote at EOF. Writing remaining buffer.")
            outfile.write(buffer + '\n')

#endregion
#region DATA CLEANING UTILITIES
####################################################################################################
# DATA CLEANING UTILITIES
####################################################################################################
def normalize_cols(data: dict) -> dict:
    """
    Normalize the keys of the dictionary to lowercase.
    1. lower the keys
    2. remove spaces from keys (replace with underscores)
    3. strip stray whitespace/control characters
    """
    normalized_data = {}
    for key, value in data.items():
        if key is None:
            continue
        new_key = key.strip().lower().replace(' ', '_').replace('\ufeff', '')  # Remove BOM
        normalized_data[new_key] = value
    return normalized_data

def check_keys(raw_qa: str, running_total: int = 0, allowed_keys={"question", "answer"}) -> tuple[bool, int]:
    # Only match keys that look like valid identifiers: "key":
    found_keys = set(re.findall(r'"(\w+)"\s*:', raw_qa))

    extra_keys = found_keys - allowed_keys
    missing_keys = allowed_keys - found_keys

    if extra_keys or missing_keys:
        running_total += 1
        return False, running_total

    return True, running_total

def parse_qa_list_TSQA(data: dict, running_total: int) -> tuple[Union[dict, None], int]:
    raw_qa = data.get("qa_list")
    is_valid, running_total = check_keys(raw_qa, running_total)
    if not is_valid:
        return None, running_total

    flags = re.DOTALL

    question_match = re.search(r'"question"\s*:\s*(.+?)"answer"\s*:', raw_qa, flags)
    answer_match = re.search(r'"answer"\s*:\s*(.+)$', raw_qa, flags)

    if not question_match or not answer_match:
        print("[ERROR] Could not extract question or answer")
        print(f"Raw: {raw_qa[:500]}")
        running_total += 1
        return data, running_total

    question = question_match.group(1).strip().strip('",}')
    answer = answer_match.group(1).strip().strip('",}')
    data["question"] = question
    data["answer"] = answer
    data = {k: v for k, v in data.items() if k != "qa_list"}

    return data, running_total

def assign_idx_key(data: list, idx_key: str = "idx") -> list:
    """
    Assign an index key to each item in the list if it doesn't already exist.
    """
    if idx_key in data[0]:
        raise ValueError(f"Key '{idx_key}' already exists in the data.")
    for i, item in enumerate(data):
        if idx_key not in item:
            item[idx_key] = f"#{i}"
    return data

def assign_idx_key(data: list, idx_key: str = "idx") -> list:
    """
    Assign an index key to each item in the list if it doesn't already exist.
    """
    if idx_key in data[0]:
        raise ValueError(f"Key '{idx_key}' already exists in the data.")
    for i, item in enumerate(data):
        if idx_key not in item:
            item[idx_key] = f"#{i}"
    return data

#endregion
#region ORGANIZE LATER
####################################################################################################
# ORGANIZE LATER
####################################################################################################
def save_output(args: argparse.Namespace, data: dict):
    """
    Append a dictionary to the specified output JSONL file.
    Creates parent directories if needed.
    """
    output_file = args.output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')
    
    logger = MasterLogger.get_instance()
    logger.info(f"Output saved to {output_file}")
