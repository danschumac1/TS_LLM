'''
2025-10-22
Author: Dan Schumacher
How to run:
   python ./src/_scratch_pad.py
'''
import numpy as np
def main():
    X_train = np.load('./data/datasets/classification/har/train.npy')
    y_train = np.load('./data/datasets/classification/har/train_labels.npy')
    X_test = np.load('./data/datasets/classification/har/test.npy')
    y_test = np.load('./data/datasets/classification/har/test_labels.npy')
    # print shapes
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # print(len(X_train))
    # print(X_train[0])
    # print(len(y_train))
    # print(y_train[0])

    # print(len(X_test))
    # print(X_test[0])
    # print(len(y_test))
    # print(y_test[0])



if __name__ == "__main__":
    main()