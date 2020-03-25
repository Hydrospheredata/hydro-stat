import datetime
from multiprocessing.pool import ThreadPool
from pprint import pprint
from interpretability.interpret import interpret
import profiler
from metric_tests import continuous_stats
import numpy as np
import pandas as pd

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}


def one_test(d1, d2, name):
    # print(name)
    stats_type1, stats_type2 = tests_to_profiles.get(name, ['same', 'same'])
    s1 = profiler.get_statistic(d1, stats_type1, None, 1)
    # pprint(name)
    s2 = profiler.get_statistic(d2, stats_type2, None, 2)
    # pprint(name)
    report = continuous_stats.test(s1, s2, name, None)
    # pprint(name)

    # pprint(report)
    return report


def final_decision(full_report):
    count_pos = 0
    count_neg = 0
    # pprint(full_report)
    for key, log in full_report.items():
        # pprint(log)
        if log['status'] == 'succeeded':
            if list(log['decision']).count("there is no change") > len(log['decision']) // 2:
                log['final_decision'] = 'there is no change'
                count_pos += 1
            else:
                log['final_decision'] = 'there is a change'
                count_neg += 1
    if count_pos > count_neg:
        full_report['final_decision'] = 'there is a change'
    else:
        full_report['final_decision'] = 'there is no change'
    return full_report


def run_once(dataset1, dataset2):
    a = datetime.datetime.now()
    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal',
             'levene_mean', 'levene_median', 'levene_trimmed',
             'sign_test', 'median_test',
             'min_max', 'ks']
    full_report = {}
    pool = ThreadPool(processes=1)
    async_results = {}
    for test in tests:
        async_results[test] = pool.apply_async(one_test, (dataset1, dataset2, test))
    for test in tests:
        full_report[test] = async_results[test].get()
    # print(type(full_report))
    full_report = final_decision(full_report)
    full_report['report'] = interpret(full_report)
    b = datetime.datetime.now()
    return (b - a).microseconds


def generate_dataset(n, f):
    return np.random.rand(n, f), np.random.rand(n, f)


def plots():
    data = pd.read_csv('times.csv')


if __name__ == '__main__':
    nb_samples = [1000000]
    nb_features = [10000]
    times = []
    for n in nb_samples:
        for f in nb_features:
            d = generate_dataset(n, f)
            t = run_once(d, d)
            times.append([n, f, t])
    pd.DataFrame(np.array(times), columns=['nb_samples', 'nb_features', 'time']).to_csv('times_big.csv', index=False)
