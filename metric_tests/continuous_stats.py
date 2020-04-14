from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from scipy.spatial import Delaunay
import numpy as np
from metric_tests.kuiper import kuiper_two, a_distance_two


def fall_in(ss1, ss2):
    smaller = (ss2 < ss1[0]).sum()
    bigger = (ss2 > ss1[1]).sum()
    return (smaller + bigger) / len(ss2)


def out_hull(inliers, outliers):
    def in_hull(p, hull):
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0

    return 1 - (in_hull(outliers, inliers).sum() / len(outliers))


def fix_shapes(data, test_type):
    if test_type in ['one_sample_t_test', 'sign_test', 'min_max']:
        return np.array(data, dtype=np.float64)
    elif test_type == 'hull':
        return data
    else:
        if len(data.shape) == 1:
            return np.array(data, dtype=np.float64).reshape(-1, 1)
        elif len(data.shape) == 2:
            return np.array(data, dtype=np.float64)


def test(s1, s2, test_type, config=None):
    report = {}
    try:
        s1 = fix_shapes(s1, test_type)
        s2 = fix_shapes(s2, test_type)

        if test_type == 'two_sample_t_test':
            results = stats.ttest_ind(s1, s2, equal_var=False)
            report['metric'] = list(results[0])
            report['p_value'] = list(results[1])
        elif test_type == 'one_sample_t_test':
            results = stats.ttest_1samp(s2, s1)
            report['metric'] = list(results[0])
            report['p_value'] = list(results[1])
        elif test_type == 'anova':
            results = stats.f_oneway(s1, s2)
            report['metric'] = list(results[0])
            report['p_value'] = list(results[1])
        elif test_type == 'mann':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.mstats.mannwhitneyu(ss1, ss2)
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'kruskal':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.mstats.kruskal(ss1, ss2)
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'brunner':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.brunnermunzel(ss1, ss2)
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'levene_mean':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.levene(ss1, ss2, center='mean')
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'levene_median':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.levene(ss1, ss2, center='median')
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'levene_trimmed':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                if config:
                    results = stats.levene(ss1, ss2, center='trimmed', proportiontocut=config.proportiontocut)
                else:
                    results = stats.levene(ss1, ss2, center='trimmed', proportiontocut=0.01)

                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'sign_test':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = sign_test(ss2, ss1)
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'median_test':
            metrics = []
            p_values = []
            medians = []
            contingency_tables = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.median_test(ss1, ss2)
                metrics.append(results[0])
                p_values.append(results[1])
                medians.append(results[2])
                contingency_tables.append(results[3])
            report['metric'] = metrics
            report['p_value'] = p_values
            if not isinstance(medians, list):
                report['median'] = medians.tolist()
            else:
                report['median'] = medians

            if not isinstance(contingency_tables, list):
                report['contingency_table'] = contingency_tables.tolist()
            else:
                report['contingency_table'] = contingency_tables
        elif test_type == 'min_max':
            metrics = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = fall_in(ss1, ss2)
                metrics.append(results)
            report['metric'] = metrics
            report['decision'] = ['there is a change' if ratio > 0.05 else 'there is no change' for ratio in
                                  report['metric']]
        elif test_type == 'ks':
            metrics = []
            p_values = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = stats.ks_2samp(ss1, ss2)
                metrics.append(results[0])
                p_values.append(results[1])
            report['metric'] = metrics
            report['p_value'] = p_values
        elif test_type == 'hull':
            results = out_hull(s1, s2)
            report['metric'] = results
            report['decision'] = 'there is a change' if results < 0.5 else 'there is no change'
        elif test_type == 'kupier':
            metrics = []
            fpps = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = kuiper_two(ss1, ss2)
                metrics.append(results[0])
                fpps.append(results[1])
                if np.isnan(results[1]):
                    raise ValueError
            report['metric'] = metrics
            report['fpp'] = fpps
            report['decision'] = ['there is a change' if ratio > 0.4 else 'there is no change' for ratio in
                                  report['metric']]
        elif test_type == 'a_dist':
            metrics = []
            for ss1, ss2 in zip(s1.T, s2.T):
                results = a_distance_two(ss1, ss2)
                metrics.append(results)
            report['metric'] = metrics
            report['decision'] = ['there is a change' if ratio > 0.4 else 'there is no change' for ratio in
                                  report['metric']]

        if 'p_value' in report.keys():
            report['decision'] = ['there is a change' if p < 0.01 else 'there is no change' for p in report['p_value']]

        if not isinstance(report.get('p_value', []), list):
            report['p_value'] = report['p_value'].tolist()

        if not isinstance(report['metric'], list):
            report['metric'] = report['metric'].tolist()
        report['status'] = 'succeeded'

    except:
        report['status'] = 'failed'

    return report
