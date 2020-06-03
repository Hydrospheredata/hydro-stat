import copy

import numpy as np

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}


def per_statistic_change_probability(tests):
    test_to_stat = {'two_sample_t_test': 'mean',
                    'one_sample_t_test': 'mean',
                    'anova': 'mean',
                    'mann': 'mean',
                    'kruskal': 'mean',
                    'levene_mean': 'std',
                    'levene_median': 'std',
                    'levene_trimmed': 'std',
                    'sign_test': 'median',
                    'median_test': 'median',
                    'ks': 'general'}

    # TODO rename variable
    x = range(len(tests[list(tests.keys())[0]]["decision"]))
    probability = [copy.deepcopy({'mean': 0, 'std': 0, 'median': 0, 'general': 0}) for _ in x]
    count = [copy.deepcopy({'mean': 0, 'std': 0, 'median': 0, 'general': 0}) for _ in x]

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
