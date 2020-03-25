import os
import zipfile

import numpy as np
from urllib import request
import gzip
import pickle
from os.path import isfile
from os import remove
import cv2
from PIL import Image
from skimage.io import imread_collection
from shutil import rmtree
import zipfile

filename = [
    'Latin.zip', 'Cyrillic.zip'
]
saved_pkl = "data/comnist.pkl"


def download_comnist():
    base_url = "https://github.com/GregVial/CoMNIST/raw/master/images/"
    if not os.path.exists('Latin.zip'):
        print('downloading Latin')
        request.urlretrieve(base_url + 'Latin.zip', 'Latin.zip')
    if not os.path.exists('Cyrillic.zip'):
        print('downloading Cyrillic')
        request.urlretrieve(base_url + 'Cyrillic.zip', 'Cyrillic.zip')
    print("Download complete.")


def fs_tree_to_dict(path_):
    file_token = ''
    for root, dirs, files in os.walk(path_):
        tree = {d: fs_tree_to_dict(os.path.join(root, d)) for d in dirs}
        tree.update({f: file_token for f in files})
        return tree


def save_comnist():
    comnist = {'Latin.zip': {'data': [], 'label': []}, 'Cyrillic.zip': {'data': [], 'label': []}}
    for name in filename:
        with zipfile.ZipFile(name) as z:
            z.extractall()
            data = fs_tree_to_dict(name.replace('.zip', ''))
            for label in data:
                print(label)
                for image_file in data[label]:
                    im = cv2.imread(name.replace('.zip', '') + '/' + label + '/' + image_file)
                    comnist[name]['data'].append(im)
                    comnist[name]['label'].append(label)
            comnist[name]['data'] = np.array(comnist[name]['data'])
            comnist[name]['label'] = np.array(comnist[name]['label'])

    with open(saved_pkl, 'wb') as f:
        pickle.dump(comnist, f)
    print("Save complete.")


def delete_comnist():
    for name in filename:
        if os.path.exists(name):
            remove(name)
            rmtree(name.replace('.zip', ''))
    print('Delete complete')


def init():
    if not isfile(saved_pkl):
        download_comnist()
        save_comnist()
        delete_comnist()
    else:
        print('comnist already downloaded')


def load(type: str):
    with open(saved_pkl, 'rb') as f:
        comnist = pickle.load(f)
    if type == 'latin':
        comnist = comnist['Latin.zip']
    if type == 'cyrillic':
        comnist = comnist['Cyrillic.zip']
    X, y = comnist['data'], comnist['label']
    return comnist


if __name__ == '__main__':
    init()
    print(load('cyrillic'))
