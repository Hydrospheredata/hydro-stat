import sys

from sklearn.model_selection import train_test_split

from utils.utils import fix_path

path = fix_path()
root = path.replace('metrics_evaluation', '')
print(root)
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([root, root + 'metrics_evaluation/CODE',
                 root + 'metrics_evaluation/CODE/libsvm/python',
                 root])

import argparse
import logging
from pprint import pprint
import numpy as np
import dataloader
import profiler
from metric_tests import continuous_stats
from metric_tests import discrete_stats
from metric_tests import image_stats
from metric_tests import time_series_stats
import reporter


def main(config):
    # setup logging
    folder = 'outputs/' + config.data_type + '/'
    file_name = config.log_name
    path = folder + file_name
    logging.basicConfig(filename=path + '.log', filemode='a', level=logging.INFO,
                        format='%(name)s -- %(asctime)s -- %(message)s')
    tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                         'min_max': ('min_max', 'same'),
                         'hull': ('delaunay', 'same')}
    image_features = ['brisque', 'brightness', 'sharpness', 'cpbd_metric', 'image_colorfulness', 'contrast',
                      'rms_contrast']

    # read datasets or generate them.
    if config.read:
        d1, d2 = dataloader.read_datasets(config.file1, config.file2)
    else:
        d1, d2 = dataloader.generate_data(config.file1, config.type)
    # print('here')

    if config.data_type == 'Image':
        D1 = None
        D2 = None
        # print('here')

        for feature in image_features:
            print(feature)
            if D1 is None:
                D1 = profiler.get_statistic(d1, feature, config, 1).reshape(-1, 1)
                D2 = profiler.get_statistic(d2, feature, config, 2).reshape(-1, 1)
                print(D1.shape)
            else:
                D1 = np.concatenate((D1, profiler.get_statistic(d1, feature, config, 1).reshape(-1, 1)),
                                    axis=1)
                D2 = np.concatenate((D2, profiler.get_statistic(d2, feature, config, 2).reshape(-1, 1)),
                                    axis=1)
                print(D1.shape)
        D1, _ = train_test_split(D1, test_size=0.33)
        D2, _ = train_test_split(D2, test_size=0.33)
        d1 = D1
        d2 = D2
        print(D1.shape)
        print(D2.shape)

    stats_type1, stats_type2 = tests_to_profiles.get(config.test_type, ['same', 'same'])
    # print('here')
    # get stats using profiler
    s1 = profiler.get_statistic(d1, stats_type1, config, 1)
    s2 = profiler.get_statistic(d2, stats_type2, config, 2)
    # compare using different metric_tests
    # print(s1)
    # print(s2)
    if config.data_type == 'Continuous':
        report = continuous_stats.test(s1, s2, config.test_type, config)
    elif config.data_type == 'Discrete':
        report = discrete_stats.test(s1, s2, config.test_type, config)
    elif config.data_type == 'Image':
        report = image_stats.perform_test(s1, s2, config.test_type, config)
    elif config.data_type == 'Text':
        report = time_series_stats.test(s1, s2, config.test_type, config)
    pprint(report)
    reporter.save_report(config, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datasets params
    parser.add_argument('--file1', type=str, default='svhn', choices=['mnist', 'svhn', 'usps', 'iris', 'wine',
                                                                      'wing_nut', 'two_diamonds', 'tetra',
                                                                      'target', 'bike_weather', 'bike_weekend',
                                                                      'lsun', 'Hepta', 'golf_ball',
                                                                      'engy_time', 'bike_season',
                                                                      'chainlink', 'atom', 'bike_weather',
                                                                      'bike_holiday',
                                                                      'bike_weekend', 'bike_season',
                                                                      'abalone_rings', 'abalone_sex',
                                                                      'auto_cylinders',
                                                                      'auto_origin', 'absenteeism_reason',
                                                                      'absenteeism_month',
                                                                      'absenteeism_day', 'absenteeism_season',
                                                                      'absenteeism_kids',
                                                                      'absenteeism_alcohol',
                                                                      'absenteeism_smoking',
                                                                      'absenteeism_pet',
                                                                      'backnote'
                                                                      ])
    parser.add_argument('--file2', type=str, default='svhn', choices=['mnist', 'svhn', 'usps', 'iris', 'wine',
                                                                      'wing_nut', 'two_diamonds', 'tetra',
                                                                      'target', 'bike_weather', 'bike_weekend',
                                                                      'lsun', 'Hepta', 'golf_ball',
                                                                      'engy_time', 'bike_season',
                                                                      'chainlink', 'atom', 'bike_weather',
                                                                      'bike_holiday',
                                                                      'bike_weekend', 'bike_season',
                                                                      'abalone_rings', 'abalone_sex',
                                                                      'auto_cylinders',
                                                                      'auto_origin', 'absenteeism_reason',
                                                                      'absenteeism_month',
                                                                      'absenteeism_day', 'absenteeism_season',
                                                                      'absenteeism_kids',
                                                                      'absenteeism_alcohol', 'absenteeism_smoking',
                                                                      'absenteeism_pet',
                                                                      'backnote'])
    parser.add_argument('--type', type=str, default='class',
                        choices=['class', 'cluster_removal', 'test_split'])
    parser.add_argument('--read', type=bool, default=False)
    # stats params
    parser.add_argument('--stats_type1', type=str, default='same',
                        choices=['mean', 'same', 'median', 'min_max', 'delaunay', 'brisque', 'brightness', 'sharpness',
                                 'cpbd_metric', 'image_colorfulness', 'contrast', 'rms_contrast', 'dominant_colors'])
    parser.add_argument('--stats_type2', type=str, default='same',
                        choices=['mean', 'same', 'median', 'min_max', 'delaunay', 'brisque', 'brightness', 'sharpness',
                                 'cpbd_metric', 'image_colorfulness', 'contrast', 'rms_contrast', 'dominant_colors'])
    parser.add_argument('--test_type', type=str, default='two_sample_t_test',
                        choices=['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal', 'brunner',
                                 'levene_mean', 'levene_median', 'levene_trimmed',
                                 'sign_test', 'median_test',
                                 'min_max', 'hull',
                                 'ks', 'kupier', 'a_dist'])
    # logging
    parser.add_argument('--data_type', type=str, default='Continuous')
    parser.add_argument('--log_name', type=str, default='image')
    # test specific parameters
    parser.add_argument('--proportiontocut', type=float, default=0.01)

    config = parser.parse_args()
    print(config)
    main(config)
