import numpy as np
from messytables import CSVTableSet, type_guess, \
    types_processor, headers_guess, headers_processor, \
    offset_processor, any_tableset
from messytables.types import *
import random
import os


def interpret(per_feature):
    report = []
    for name, value in per_feature.items():
        if value['drift-probability'] > 0.5:
            message = 'the feature "{}" has changed.'.format(name)
            report.append({'message': message, 'drift_probability_per_feature': value['drift-probability']})

    return report


def get_types(file, delimiter=None):
    r1 = random.randint(0, 10000)
    file.to_csv(str(r1) + '.csv', index=False)
    fh = open(str(r1) + '.csv', 'rb')
    table_set = CSVTableSet(fh)
    row_set = table_set.tables[0]
    offset, headers = headers_guess(row_set.sample)
    row_set.register_processor(headers_processor(headers))
    row_set.register_processor(offset_processor(offset + 1))
    types = type_guess(row_set.sample, strict=True)
    os.remove(str(r1) + '.csv')
    return headers, types


def get_cont(data, types):
    result = None
    f = []

    for i, t in enumerate(types[1]):
        t = str(t)
        if t in ('Integer', 'Float', 'Decimal'):
            if result is None:
                result = data[:, i].reshape(-1, 1)
            else:
                result = np.append(result, data[:, i].reshape(-1, 1), axis=1)
            f.append(types[0][i])
    result = np.array(result, dtype=np.float64)
    return result, f


def get_disc(data, types):
    result = None
    f = []
    for i, t in enumerate(types[1]):
        t = str(t)
        if t == 'String':
            if result is None:
                result = data[:, i].reshape(-1, 1)
            else:
                result = np.append(result, data[:, i].reshape(-1, 1), axis=1)
            f.append(types[0][i])
    result = np.array(result)
    return result, f
