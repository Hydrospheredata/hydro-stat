import os

import numpy as np
from urllib import request
import gzip
import pickle
from os.path import isfile
from os import remove
from scipy import io
import subprocess

filename = [
    ["train", "train_32x32.mat"],
    ["test", "test_32x32.mat"],
]
saved_pkl = "data/pacs.pkl"


def download_pacs():
    base_url = 'https://doc-0k-ac-docs.googleusercontent.com/docs/securesc/36kjemri03jt25mtkcrksg0vhes4n6qu/c68flnraqdoq4q7hcsbdmja9thl0vpdh/1578909600000/15518465043025350564/15518465043025350564/1jFpmdUnM5xZUFtpOSDolznPsK9VDi1vf?e=download&authuser=0'
    name = "pacs.zip"
    print("Downloading usps dataset...")
    if not os.path.exists(name):
        os.system('wget '+base_url)
        subprocess.run(["wget", base_url])
        request.urlretrieve(base_url, name)
    exit()
    print("Download complete.")


def save_pacs():
    name = "pacs.pkl"
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


def delete_pacs():
    name = "pacs.pkl"
    if os.path.exists(name[1]):
        remove(name[1])

    print('Delete complete')


def init():
    if not isfile(saved_pkl):
        download_pacs()
        save_pacs()
        delete_pacs()
    else:
        print('usps already downloaded')


def load():
    with open(saved_pkl, 'rb') as f:
        usps = pickle.load(f)
    return usps["training_images"], usps["training_labels"], usps["test_images"], usps["test_labels"]


if __name__ == '__main__':
    init()
