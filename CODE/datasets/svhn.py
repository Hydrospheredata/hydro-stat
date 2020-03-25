import os
import random

import numpy as np
from urllib import request
import gzip
import pickle
from os.path import isfile
from os import remove
from scipy import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

filename = [
    ["train", "train_32x32.mat"],
    ["test", "test_32x32.mat"],
]
saved_pkl = "data/svhn.pkl"


def download_svhn():
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        if not os.path.exists(name[1]):
            request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def fix_svhn(X):
    # print(X.shape)
    newX = np.zeros((X.shape[-1], 28, 28, 3), dtype=np.uint8)
    for i in range(X.shape[-1]):
        newX_i = X[:, :, :, i]
        newX_i = cv2.resize(newX_i, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        # print(newX_i.shape)
        newX[i] = newX_i
    # print(newX.shape)
    return newX


def save_svhn():
    svhn = {}
    train = io.loadmat(filename[0][1])
    test = io.loadmat(filename[1][1])
    svhn['training_images'] = fix_svhn(train['X'])
    svhn['training_labels'] = train['y']
    svhn['test_images'] = fix_svhn(test['X'])
    svhn['test_labels'] = test['y']

    with open(saved_pkl, 'wb') as f:
        pickle.dump(svhn, f)
    print("Save complete.")


def delete_svhn():
    for name in filename:
        if os.path.exists(name[1]):
            remove(name[1])
    print('Delete complete')


def init():
    if not isfile(saved_pkl):
        download_svhn()
        save_svhn()
        delete_svhn()
    else:
        print('svhn already downloaded')


def load():
    with open(saved_pkl, 'rb') as f:
        svhn = pickle.load(f)
    return svhn["training_images"], svhn["training_labels"], svhn["test_images"], svhn["test_labels"]


if __name__ == '__main__':
    init()
