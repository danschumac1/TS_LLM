'''
2025-09-09
Author: Dan Schumacher
How to run:
   python ./src/_scratchpad.py
'''

from utils.file_io import load_jsonl

def main():
    mcq1 = load_jsonl("./data/datasets/MCQ1/train.jsonl")
    timerbed = load_jsonl("./data/datasets/TimerBed/train.jsonl")

    for dataset_name, dataset in [("MCQ1", mcq1), ("TimerBed", timerbed)]:
        print(f"\n=== {dataset_name} Sample ===\n")
        for i, entry in enumerate(dataset[:1], 1):
            print(f"--- Entry {i} ---")
            for k, v in entry.items():
                if k == "ts":
                    print(f"{k}:")
                    for dim, lst in v.items():
                        print(f"  {dim}: {lst}")
                else:
                    print(f"{k}: {v}")
            print()  # blank line between entries


if __name__ == "__main__":
    main()