import itertools
import logging
import os

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def name(file):
    return str(file).split('/')[-1].split('.')[0].replace(' ', '_')


def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname


def plot(x, config=None, file=1, save_overlap=True):
    logger = logging.getLogger('plots_logger')
    if file == 1:
        n, bins, patches = plt.hist(x, 20, density=True, alpha=0.75, label='training')
        plt.legend()
        plt.xlabel('Brisque scores')
        plt.ylabel('Probability')
        saving_file = name(config.file1)
        stats = str(config.stats_type2)
        plt.title('Histogram of Brisque Score for ' + saving_file)
        plt.grid(True)
        file_name = unique_file('graphs/' + saving_file + '__' + stats, 'png')
        # plt.savefig(file_name)
        logger.info(
            'data:{}, plot_file_name:{}'.format(saving_file, file_name))
    else:
        n, bins, patches = plt.hist(x, 20, density=True, alpha=0.75, label='deployment')
        plt.legend()
        plt.xlabel('Brisque scores')
        plt.ylabel('Probability')
        saving_file = name(config.file2)
        if saving_file == '':
            saving_file = "generated_" + name(config.file1)
        stats = str(config.stats_type2)
        if save_overlap:
            saving_file += '_overlap'
        plt.title('Histogram of Brisque Score for ' + saving_file)
        plt.grid(True)
        file_name = unique_file('graphs/' + saving_file + '__' + stats, 'png')
        # plt.savefig(file_name)
        logger.info(
            'data:{}, plot_file_name:{}'.format(saving_file, file_name))
        if save_overlap:
            plt.show()
            plot(x, config=config, file=file, save_overlap=False)
        else:
            plt.show()


def converter(s1, func):
    if isinstance(s1, tuple):
        x, y = s1
    else:
        x = s1
    values = []
    for img in x:
        # print("not here")
        try:
            values.append(func(img))
        except:
            pass

    return np.array(values)


def get_statistic(data, stat_type, config=None, file=1):
    stat = data
    if stat_type == "mean":
        stat = np.mean(data, axis=0)
    elif stat_type == 'same':
        stat = data
    elif stat_type == 'median':
        stat = np.median(data, axis=0)
    elif stat_type == 'min_max':
        stat = np.vstack((np.min(data, axis=0), np.max(data, axis=0)))
    elif stat_type == 'delaunay':
        stat = Delaunay(data)
    return stat
