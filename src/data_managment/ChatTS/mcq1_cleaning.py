'''
2025-08-29
Author: Dan Schumacher
How to run:
   python ./src/data_managment/ChatTS/mcq1_cleaning.py
'''

import os
import sys
sys.path.append("./src")
import random

from utils.file_io import load_jsonl, save_jsonl

def main():
    data = load_jsonl("data/datasets/_raw_data/ChatTS/TSandLanguage/MCQ_1_TS.jsonl")
    
    output_data = []
    for idx, row in enumerate(data):
        line_dict = {}
        line_dict["idx"] = f"IDX_{idx}"
        
        # build option string
        abcd_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J"}

        options_list = [f"[{abcd_map[i]}] {opt}" for i, opt in enumerate(row["options"])]
        option_str = "Your options are:\n\t" + "\n\t".join(options_list)

        # full question with description + question + options
        question = f"{row['description']}\n\n{row['question']}\n\n{option_str}"
        line_dict["question"] = question

        # answer as string
        answer = row["options"][int(row["answer_index"])]
        line_dict["answer"] = answer
        
        # MISC
        line_dict["label"] = abcd_map[int(row["answer_index"])]
        line_dict["dataset"] = "ChatTS"
        line_dict["split"] = "MCQ_1"
        line_dict["task_type"] = "MCQ"
        line_dict["multivariate"] = False
        line_dict["uuid"] = row["uuid"]

        # time series
        line_dict["ts"] = {"dim_0": row["series"]}

        output_data.append(line_dict)

    # sample 1000 rows
    random.seed(42)
    random.shuffle(output_data)
    train_out = output_data[0:1500]
    dev_out = output_data[1500:2500]
    test_out = output_data[2500:3500]

    out_dir = "./data/datasets/MCQ1"
    os.makedirs(out_dir, exist_ok=True)

    for name, out in {"train": train_out, "dev": dev_out, "test": test_out}.items():
        save_jsonl(out, f"{out_dir}/{name}.jsonl")
    print("save successful")

if __name__ == "__main__":
    main()