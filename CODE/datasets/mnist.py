import os

import numpy as np
from urllib import request
import gzip
import pickle
from os.path import isfile
from os import remove

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]
saved_pkl = "data/mnist.pkl"


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        if not os.path.exists(name[1]):
            request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(saved_pkl, 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def delete_mnist():
    for name in filename:
        if os.path.exists(name[1]):
            remove(name[1])
    print('Delete complete')


def init():
    if not isfile(saved_pkl):
        download_mnist()
        save_mnist()
        delete_mnist()
    else:
        print('mnist already downloaded')


def load():
    with open(saved_pkl, 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


if __name__ == '__main__':
    init()
