'''
2025-09-29
Author: Dan Schumacher
How to run:
   python ./src/data_managment/create_tiny_test.py
'''

import sys, os, random
from typing import List
sys.path.append("./src/")

from utils.file_io import load_jsonl, save_jsonl

def make_sample(dataset:str="TimerBed",split:str="test") -> List[dict]:
    data_dict = {}
    for subset in os.listdir(f"./data/datasets/{dataset}"):
        full_path = os.path.join(f"./data/datasets/{dataset}", subset)

        if not os.path.isdir(full_path):
            continue

        for filename in os.listdir(full_path):
            if split in filename:
                file_path = os.path.join(full_path, filename)
                data_dict[subset] = load_jsonl(file_path)

                print(f"Loaded {len(data_dict[subset])} rows from {file_path}")
                continue

    flat_data = []

    for subset, data in data_dict.items():
        samp = random.sample(data, 15)
        for line in samp:
            line["subset"] = subset
        flat_data.extend(samp)
    
    random.shuffle(flat_data)
    print(f"Made tiny dataset with {len(flat_data)} rows")

    save_path = f"./data/datasets/{dataset}/_TINYTEST"
    print(f"Saving to {save_path}/{split}.jsonl")
    os.makedirs(save_path, exist_ok=True)
    save_jsonl(flat_data, f"{save_path}/{split}.jsonl")

def main():
    make_sample("TimerBed", "test")
    make_sample("TimerBed", "train")
    





if __name__ == "__main__":
    main()