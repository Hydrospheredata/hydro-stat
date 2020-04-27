import numpy as np
from scipy.spatial import Delaunay


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
