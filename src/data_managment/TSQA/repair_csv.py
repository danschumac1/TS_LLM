'''
2025-08-07
Author: Dan Schumacher
How to run:
   python ./src/data_managment/repair_csv.py
'''

import sys
sys.path.append("./src")
from utils.data_utils import fix_multiline_csv

def main():
    pre_path = "./data/datasets/raw_data/TSQA/open_ended_qa/"
    # fix_multiline_csv(f"{pre_path}/forecasting_imputation1_broken.csv", f"{pre_path}/forecasting_imputation1.csv")
    # fix_multiline_csv(f"{pre_path}/forecasting_imputation2_broken.csv", f"{pre_path}/forecasting_imputation2.csv")
    fix_multiline_csv(f"{pre_path}/open_ended_qa_broken.csv", f"{pre_path}/open_ended_qa.csv")


if __name__ == "__main__":
    main()