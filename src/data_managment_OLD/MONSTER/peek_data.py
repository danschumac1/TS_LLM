'''
2025-10-10
Author: Dan Schumacher
How to run:
   python ./src/data_managment/MONSTER/peek_data.py
'''
import numpy as np

def main():

    # load the array
    X = np.load("./data/datasets/_raw_data/MONSTER/Pedestrian_X.npy", mmap_mode="r")  # memory-map so it won’t blow up RAM

    # basic info
    print("Type:", type(X))
    print("Shape:", X.shape)
    print("Dtype:", X.dtype)

    # peek at a small slice
    print("First row:\n", X[0])
    print("First 5 rows:\n", X[:5])


if __name__ == "__main__":
    main()