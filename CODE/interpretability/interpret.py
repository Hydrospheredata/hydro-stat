import numpy as np
from messytables import CSVTableSet, type_guess, \
    types_processor, headers_guess, headers_processor, \
    offset_processor, any_tableset
from messytables.types import *


def interpret(per_feature):
    report = []
    for name, value in per_feature.items():
        if value['drift-probability'] > 0.5:
            message = 'the feature "{}" has changed.'.format(name)
            report.append({'message': message, 'drift_probability_per_feature': value['drift-probability']})
        # nb_features = -1
        # successful_tests = 0
        # for test, result in report.items():
        #     if test != 'final_decision':
        #         if result['status'] == 'succeeded':
        #             successful_tests += 1
        #             if nb_features == -1:
        #                 nb_features = len(result['metric'])
        #                 feature_changes = [0 for i in range(nb_features)]
        #             decisions = result['decision']
        #             for i, decision in enumerate(decisions):
        #                 if decision == 'there is a change':
        #                     feature_changes[i] += 1
        #
        # feature_changes = np.array(feature_changes) / successful_tests
        # report = []
        # for i, feature in enumerate(feature_changes):
        #     if feature > 0.5:
        #         feature_report = '- feature={0} has changed with a probability of {1:.2f}'.format(i, feature)
        #         report.append(feature_report)

    return report


def get_types(file, delimiter=None):
    fh = open(file, 'rb')

    # Load a file object:
    table_set = CSVTableSet(fh)

    # If you aren't sure what kind of file it is, you can use
    # any_tableset.
    # table_set = any_tableset(fh)

    # A table set is a collection of tables:
    row_set = table_set.tables[0]
    # print(row_set.sample)
    # for item in  row_set:
    #     print(item)

    # guess header names and the offset of the header:
    offset, headers = headers_guess(row_set.sample)
    row_set.register_processor(headers_processor(headers))
    print(headers)
    # add one to begin with content, not the header:
    row_set.register_processor(offset_processor(offset + 1))

    # guess column types:
    types = type_guess(row_set.sample, strict=True)
    print(types)
    return headers, types


def get_cont(data, types):
    result = None
    print(data.dtype)
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
    print(result.dtype)
    return result, f


def get_disc(data, types):
    result = None
    print(data.dtype)
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
    print(result.dtype)
    return result, f
