import numpy as np

from sklearn import preprocessing
from scipy.stats import entropy

from hydro_stat import continious


def nominal_histogram(data):
    key, count = np.unique(data, return_counts=True)
    return {k: c for k, c in zip(key, count)}


def maximal_difference(s1, s2):
    max_dif = 0
    count = 0
    elements = 0
    for k1 in s1.keys():
        elements += s2.get(k1, 0) + s1.get(k1, 0)
        count += 2
        dif = np.abs(s2.get(k1, 0) - s1.get(k1, 0))
        if dif > max_dif:
            max_dif = dif
    for k2 in s2.keys():
        count += 2
        elements += s2.get(k1, 0) + s1.get(k1, 0)
        dif = np.abs(s2.get(k2, 0) - s1.get(k2, 0))
        if dif > max_dif:
            max_dif = dif
    return min(1., max_dif / (elements / count))


def fall_into(s1, s2):
    vfunc = np.vectorize(lambda x: x in s1)
    return max(min((len(s2) - vfunc(s2).sum()) / len(s2), 1.), 0.0)


def ordinal_ks(ordinal1, ordinal2, mapping=None):
    if mapping == None:
        le = preprocessing.LabelEncoder()
        le.fit(list(ordinal1) + list(ordinal2))
        ordering = lambda t: le.transform([t])[0]
    else:
        ordering = lambda t: mapping[t]
    vfunc = np.vectorize(ordering)
    return continious.test(vfunc(ordinal1), vfunc(ordinal2), 'kruskal')


def clean(param):
    data = []
    for p in param:
        for d in p:
            data.append(d)
    return np.array(data).reshape(-1, 1)


def fix(hist1, hist2):
    keys = list(set(hist2.keys()).union(set(hist1.keys())))
    counts1 = []
    counts2 = []
    for key in keys:
        counts1.append(hist1.get(key, 0))
        counts2.append(hist2.get(key, 0))
    return keys, counts1, counts2


def process_feature(training_data, production_data):
    hist1 = nominal_histogram(training_data)
    hist2 = nominal_histogram(production_data)
    bins, train, dep = fix(hist1, hist2)

    histogram = {'bins': bins, 'deployment': dep, 'training': train}
    stats = {
        'entropy': {
            'training': entropy(list(hist1.values()), base=2),
            'deployment': entropy(list(hist2.values()), base=2),
            'change_probability': 1 - ordinal_ks(production_data, training_data)['p_value'][0]
        },
        'unique values': {
            'training': len(list(hist1.keys())),
            'deployment': len(list(hist2.keys())),
            'change_probability': 1 - ordinal_ks(production_data, training_data)['p_value'][0]
        }
    }

    feature = {'drift-probability': 1 - ordinal_ks(production_data, training_data)['p_value'][0], 'histogram': histogram, 'statistics': stats}
    return feature


if __name__ == '__main__':
    s1 = np.array(["sun", "moon", "shine", "boy"])
    s2 = np.array(["sun", "moon", "hello", "girl", "boy", "girl", "girl", "girl", "girl", "girl"])
