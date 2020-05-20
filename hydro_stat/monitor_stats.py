import numpy as np


def get_numerical_statistics(data):
    means = np.mean(data, axis=0)
    medians = np.median(data, axis=0)
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

    # TODO select number of bins based on config?
    training_histogram, bin_edges = np.histogram(training_data, bins=number_of_bins, normed=True, density=True,
                                                 range=[data_minimum, data_maximum])
    deployment_histogram, _ = np.histogram(deployment_data, bins=bin_edges, normed=True, density=True, range=[data_minimum, data_maximum])

    return training_histogram, bin_edges, deployment_histogram


def get_histograms(training_data, deployment_data, feature_names=None):
    histograms = {}
    for i, (t, d) in enumerate(zip(training_data.T, deployment_data.T)):
        title = feature_names[i] if feature_names is not None else i
        training_hist, bins, deployment_hist = _get_histogram(t, d)
        histograms[title] = {'training': training_hist, 'deployment': deployment_hist, 'bins': bins}
    return histograms
