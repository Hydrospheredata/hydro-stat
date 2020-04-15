import statistics

import numpy as np
from scipy.stats import entropy as ent


# import numpy as np

# TODO remove these custom functions
def mean(data):
    return np.mean(data, axis=0)


def std(data):
    return np.std(data, axis=0)


def median(data):
    return np.median(data, axis=0)


def mode(data):
    return statistics.mode(data, axis=0)


def entropy(data):
    return ent(data, axis=0)


def get_all(data):
    means = mean(data)
    medians = median(data)
    # modes = mode(data)
    stds = std(data)

    stats = {'means': means,
             'stds': stds,
             'medians': medians}
    return stats


def _get_histogram(training_data, deployment_data, number_of_bins=20):
    data_minimum = min(training_data.min(), deployment_data.min())
    data_maximum = max(training_data.max(), deployment_data.max())

    training_histogram, bin_edges = np.histogram(training_data, bins=number_of_bins, normed=True, range=[data_minimum, data_maximum])
    deployment_histogram, _ = np.histogram(deployment_data, bins=bin_edges, normed=True, range=[data_minimum, data_maximum])

    return training_histogram, bin_edges, deployment_histogram,


def get_histograms(tr, dep, f1=None, f2=None, bins__=20):
    histograms = {}
    for i, _ in enumerate(zip(tr.T, dep.T)):
        t, d = _
        title = f1[i] if f1 is not None else None
        n, bins, n_ = _get_histogram(t, d, bins__)
        histograms[title] = {'training': n, 'deployment': n_, 'bins': bins}
    return histograms
