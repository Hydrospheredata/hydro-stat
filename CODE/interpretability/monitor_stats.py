import numpy as np
import statistics
from scipy.stats import entropy as ent
# import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
    # print(data.shape)
    # print(means.shape)
    # print(medians.shape)
    # print(stds.shape)
    stats = {'means': means,
             'stds': stds,
             'medians': medians}
    return stats


def _get_histogram(t, d, bins__=20, i=0, title=None):
    r1 = min(t.min(), d.min())
    r2 = max(t.max(), d.max())
    n, bins, _ = plt.hist(t, bins=bins__, normed=True, range=[r1, r2], label='training', alpha=0.6)
    n_, _, _ = plt.hist(d, bins=bins, normed=True, range=[r1, r2], label='deployment', alpha=0.4)

    plt.title('feature {}'.format(i) if title is None else title)
    plt.legend()
    plt.savefig('graphs/feature {}.png'.format(i) if title is None else 'graphs/' + title)
    plt.show()
    return n, bins, n_,


def get_histograms(tr, dep, f1=None, f2=None, bins__=20):
    histograms = {}
    for i, _ in enumerate(zip(tr.T, dep.T)):
        t, d = _
        title = f1[i] if f1 is not None else None
        n, bins, n_ = _get_histogram(t, d, bins__, i, title)
        histograms[title] = {'training': n, 'deployment': n_, 'bins': bins}
    return histograms
