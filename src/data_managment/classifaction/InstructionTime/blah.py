'''
2025-10-22
Author: Dan Schumacher
How to run:
   python ./src/data_managment/classifaction/InstructionTime/blah.py
'''
''' INFORMATION ABOUT DATASETS:

    cpu
        univariate
        series length: 720
        num classes: 2
        num samples: 500
        ------- 
        train: arff, ts, and txt file
        test: arff, ts, and txt file

        
    whale (rcw) (Cornell Whale Challenge)
        univariate
        series length: 4000
        num classes: 2
        num samples: 30,000
        -------
        a train folder with a bunch (30,000) of aiffs
        a test folder with a bunch (54,502) of aiffs


    tee (Lightning / TEE)
        univariate
        series length: 319
        num classes: 7
        num samples: 143
        -------

        
    ecg (A MESS DON'T USE FOR NOW)
        univariate
        series length: 1500
        num classes: 4
        num samples: 43,673
        -------

        
    emg
        univariate
        series length: 1500
        num classes: 3
        num samples: 205
        ------- 
        healthy: dat, hea, and txt file
        myopathy: dat, hea, and txt file
        neuropathy: dat, hea, and txt file

        
    har
        multivariate (3 variables)
        series length: 206
        num classes: 6
        num samples: 10,299
        ------- 
        X_train.txt, y_train.txt,
        X_test.txt, y_test.txt
        also have something called Inertial Signals?
            it contains xyz of body acc, body gyro, and total acc .txts
#endregion
'''

import sys
from typing import List; sys.path.append("./src/")
import numpy as np
import os

PREPATH = "data/datasets/classification/InstructionTime/"
DATASETS=["cpu", "whale", "tee", "emg", "har"] # "ecg"


def read_arff_file(file_path):
    pass

def read_ts_file(file_path):
    pass

def read_txt_file(file_path:str) -> List[np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        str_values = line.strip().split()
        float_values = [float(val) for val in str_values]
        data.append(np.array(float_values, dtype=np.float32))
    return np.array(data)
    
def save_to_npy(data:List[np.ndarray], save_path:str):
    np.save(save_path, data)
    print(f"Data saved to {save_path}")

def main():
    data = read_txt_file("data/datasets/classification/InstructionTime/cpu/Computers_TEST.txt")
    save_to_npy(data, "quick_test.npy")
    
if __name__ == "__main__":
    main()