import sys

from sklearn.model_selection import GridSearchCV

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

import dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import reporter


def main(config):
    # setup logging
    folder = '../outputs/' + config.data_type + '/'
    file_name = config.log_name
    path = folder + file_name
    logging.basicConfig(filename=path + '.log', filemode='a', level=logging.INFO,
                        format='%(name)s -- %(asctime)s -- %(message)s')
    tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                         'min_max': ('min_max', 'same'),
                         'hull': ('delaunay', 'same')}

    # read datasets or generate them.
    # read datasets or generate them.
    if config.read:
        d1, d2 = dataloader.read_datasets(config.file1, config.file2)
    else:
        d1, d2 = dataloader.generate_data_with_labels(config.file1, config.type)

    x1, y1 = d1
    x2, y2 = d2
    print(len(x1))

    xg_boost_parameters = {'nthread': [7],
                           'learning_rate': [0.05, 0.01],
                           'max_depth': [5, 15],
                           'min_child_weight': [10],
                           'silent': [1],
                           'subsample': [0.8, 0.9],
                           'colsample_bytree': [0.7],
                           'n_estimators': [15, 50],
                           'seed': [1337]}
    tree_para = {'criterion': ['gini', 'entropy'],
                 'max_depth': [4, 10, 50]}

    knn_para = {'n_neighbors': [3, 5, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']}
    Cs = [0.001, 1, 10]
    gammas = [0.001, 0.1]
    svm_params = {'C': Cs, 'gamma': gammas}

    classifiers = {'lg': LogisticRegression(random_state=0, solver='lbfgs'),
                   'svm': GridSearchCV(SVC(gamma='auto'), svm_params, cv=5, n_jobs=8),
                   'nb': GaussianNB(),
                   'knn': GridSearchCV(KNeighborsClassifier(), knn_para, cv=5, n_jobs=8),
                   'dt': GridSearchCV(DecisionTreeClassifier(random_state=0), tree_para, cv=5, n_jobs=8),
                   'xgboost': GridSearchCV(XGBClassifier(), xg_boost_parameters, n_jobs=8, cv=5)}

    clf = classifiers[config.classifier]

    clf.fit(x1, y1)
    pred1 = clf.predict(x1)

    report_train = classification_report(y1, pred1)

    pred2 = clf.predict(x2)

    report_test = classification_report(y2, pred2)
    report = {'cls': config.classifier,
              'train': report_train,
              'test': report_test}
    pprint(report)
    reporter.save_report(config, report)

    # if config.data_type == 'Image':
    #     d1 = profiler.get_statistic(d1, config.stats_type1, config, 1)
    #     d2 = profiler.get_statistic(d2, config.stats_type2, config, 2)
    #
    # stats_type1, stats_type2 = tests_to_profiles.get(config.test_type, ['same', 'same'])
    # # print('here')
    # # get stats using profiler
    # s1 = profiler.get_statistic(d1, stats_type1, config, 1)
    # s2 = profiler.get_statistic(d2, stats_type2, config, 2)
    # # compare using different metric_tests
    # if config.data_type == 'Continuous':
    #     report = continuous_stats.test(s1, s2, config.test_type, config)
    # elif config.data_type == 'Discrete':
    #     report = discrete_stats.test(s1, s2, config.test_type, config)
    # elif config.data_type == 'Image':
    #     report = image_stats.test(s1, s2, config.test_type, config)
    # elif config.data_type == 'Text':
    #     report = time_series_stats.test(s1, s2, config.test_type, config)
    # pprint(report)
    # reporter.save_report(config, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datasets params
    parser.add_argument('--file1', type=str, default='iris',
                        choices=['mnist', 'svhn', 'usps', 'iris', 'wine',
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
    parser.add_argument('--file2', type=str, default='', choices=['mnist', 'svhn', 'usps', 'iris', 'wine',
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
    parser.add_argument('--type', type=str, default='cluster_removal',
                        choices=['cluster_removal', 'test_split'])
    parser.add_argument('--read', type=bool, default=False)

    parser.add_argument('--stats_type1', type=str, default='brightness',
                        choices=['mean', 'same', 'median', 'min_max', 'delaunay', 'brisque', 'brightness', 'sharpness',
                                 'cpbd_metric', 'image_colorfulness', 'contrast', 'rms_contrast', 'dominant_colors'])
    parser.add_argument('--stats_type2', type=str, default='brightness',
                        choices=['mean', 'same', 'median', 'min_max', 'delaunay', 'brisque', 'brightness', 'sharpness',
                                 'cpbd_metric', 'image_colorfulness', 'contrast', 'rms_contrast', 'dominant_colors'])
    parser.add_argument('--test_type', type=str, default='two_sample_t_test',
                        choices=['two_sample_t_test', 'one_sample_t_test', 'anova', 'mann', 'kruskal', 'brunner',
                                 'levene_mean', 'levene_median', 'levene_trimmed',
                                 'sign_test', 'median_test',
                                 'min_max', 'hull',
                                 'ks',
                                 ''])

    parser.add_argument('--classifier', type=str, default='xgboost',
                        choices=['lg', 'svm', 'nb', 'knn', 'dt', 'xgboost'])
    # logging
    parser.add_argument('--data_type', type=str, default='Continuous')
    parser.add_argument('--log_name', type=str, default='classifier')
    # test specific parameters
    parser.add_argument('--proportiontocut', type=float, default=0.01)

    config = parser.parse_args()
    print(config)
    main(config)
