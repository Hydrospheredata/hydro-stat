import numpy as np


# import numpy as np


def get_all(data):
    means = np.mean(data, axis=0)
    medians = np.median(data, axis=0)
    # modes = mode(data)
    stds = np.std(data, axis=0)

    stats = {'means': means,
             'stds': stds,
             'medians': medians}
    return stats


def _get_histogram(training_data, deployment_data, number_of_bins=20):
    training_data = training_data.astype(float)
    deployment_data = deployment_data.astype(float)
    data_minimum = min(training_data.min(), deployment_data.min())
    data_maximum = max(training_data.max(), deployment_data.max())
    training_histogram, bin_edges = np.histogram(training_data, bins=number_of_bins, normed=True, density=True,
                                                 range=[data_minimum, data_maximum])
    deployment_histogram, _ = np.histogram(deployment_data, bins=bin_edges, normed=True, density=True,
                                           range=[data_minimum, data_maximum])

    return training_histogram, bin_edges, deployment_histogram


def get_histograms(tr, dep, f1=None):
    histograms = {}
    for i, _ in enumerate(zip(tr.T, dep.T)):
        t, d = _
        title = f1[i] if f1 is not None else None
        n, bins, n_ = _get_histogram(t, d)
        histograms[title] = {'training': n, 'deployment': n_, 'bins': bins}
    return histograms
