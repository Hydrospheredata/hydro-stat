import os

import numpy as np
from urllib import request
import gzip
import pickle
from os.path import isfile
from os import remove
from scipy import io

filename = [
    ["train", "train_32x32.mat"],
    ["test", "test_32x32.mat"],
]
saved_pkl = "data/usps.pkl"


def download_usps():
    base_url = "https://github.com/mingyuliutw/CoGAN/raw/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"
    name = "usps.pkl"
    print("Downloading usps dataset...")
    if not os.path.exists(name):
        request.urlretrieve(base_url, name)

    print("Download complete.")


def save_usps():
    name = "usps.pkl"
    f = gzip.open(name, "rb")
    data_set = pickle.load(f, encoding="bytes")
    f.close()
    usps = {}
    train = data_set[0]
    test = data_set[1]
    usps['training_images'] = train[0]
    usps['training_labels'] = train[1]
    usps['test_images'] = test[0]
    usps['test_labels'] = test[1]

    with open(saved_pkl, 'wb') as f:
        pickle.dump(usps, f)

    print("Save complete.")


def delete_usps():
    name = "usps.pkl"
    if os.path.exists(name[1]):
        remove(name[1])

    print('Delete complete')


def init():
    if not isfile(saved_pkl):
        download_usps()
        save_usps()
        delete_usps()
    else:
        print('usps already downloaded')


def load():
    with open(saved_pkl, 'rb') as f:
        usps = pickle.load(f)
    return usps["training_images"], usps["training_labels"], usps["test_images"], usps["test_labels"]


if __name__ == '__main__':
    init()


