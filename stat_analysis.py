import copy

import numpy as np

from hydro_stat import continious, profiler

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}


def one_test(d1, d2, name):
    stats_type1, stats_type2 = tests_to_profiles.get(name, ['same', 'same'])
    s1 = profiler.get_statistic(d1, stats_type1, None, 1)
    s2 = profiler.get_statistic(d2, stats_type2, None, 2)
    report = continious.test(s1, s2, name, None)

    return report


def final_decision(full_report):
    count_pos = 0
    count_neg = 0
    for key, log in full_report.items():
        if log['status'] == 'succeeded':
            if list(log['decision']).count("there is no change") > len(log['decision']) // 2:
                log['final_decision'] = 'there is no change'
                count_pos += 1
            else:
                log['final_decision'] = 'there is a change'
                count_neg += 1
    if count_pos < count_neg:
        return 'there is a change'
    else:
        return 'there is no change'


def fix(f, stats, histograms, stats2, per_stat, per_feature):
    per_feature_report = {}
    for i, name in enumerate(f):
        histogram = histograms[name]
        stat = {}
        for statistic_name, values in stats.items():
            statistic_name = statistic_name[:-1]
            stat[statistic_name] = {}
            stat[statistic_name]['training'] = values[i]
            stat[statistic_name]['deployment'] = stats2[statistic_name + 's'][i]
            stat[statistic_name]['change_probability'] = per_stat[i][statistic_name]

        per_feature_report[name] = {"histogram": histogram, "statistics": stat,
                                    "drift-probability": per_feature[i]}
    return per_feature_report


def overall_probability_drift(tests):
    probability = 0
    count = 0
    for test_name, test in tests.items():
        if test['status'] == 'succeeded':
            for decision in test['decision']:
                if decision == 'there is a change':
                    probability += 1
                count += 1
    return -1.0 if count == 0 else probability / count


def per_feature_change_probability(tests):
    probability = [0] * len(tests[list(tests.keys())[0]]["decision"])
    count = [0] * len(tests[list(tests.keys())[0]]["decision"])

    for test_name, test in tests.items():
        if test['status'] == 'succeeded':
            for i, decision in enumerate(test['decision']):
                if decision == 'there is a change':
                    probability[i] += 1
                count[i] += 1
    return -1.0 if count == 0 else np.array(probability) / np.array(count)


def per_statistic_change_probability(tests):
    test_to_stat = {'two_sample_t_test': 'mean', 'one_sample_t_test': 'mean', 'anova': 'mean',
                    'mann': 'mean', 'kruskal': 'mean', 'levene_mean': 'std',
                    'levene_median': 'std', 'levene_trimmed': 'std',
                    'sign_test': 'median', 'median_test': 'median',
                    'ks': 'general'
                    }
    probability = [copy.deepcopy({'mean': 0, 'std': 0, 'median': 0, 'general': 0})
                   for _ in range(len(
            tests[list(tests.keys())[0]]["decision"]))]
    count = [copy.deepcopy({'mean': 0, 'std': 0, 'median': 0, 'general': 0})
             for _ in range(len(
            tests[list(tests.keys())[0]]["decision"]))]
    for test_name, test in tests.items():
        if test['status'] == 'succeeded':
            for i, decision in enumerate(test['decision']):
                stat = test_to_stat[test_name]
                if decision == 'there is a change':
                    probability[i][stat] += 1
                count[i][stat] += 1
    for p in range(len(probability)):
        for t in probability[p]:
            probability[p][t] /= count[p][t]
    return -1.0 if count == 0 else np.array(probability)
