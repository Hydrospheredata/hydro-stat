import sys

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

import numpy as np
import umap
import matplotlib.pyplot as plt
import reporter


def sammon_error(X, _X, distance_metric=lambda x1, x2: np.linalg.norm(x1 - x2)):
    '''
    computes sammon's error for original points in original space and points in reduced space
    X in Rn
    _X in Rm - reduced space
    X: points in original space
    _X: points in projecteed space
    distance_metric: Callable - f(x1, x2)-> float

    '''
    assert len(X) == len(_X)
    orig_distances = np.array([distance_metric(X[i], X[i + 1]) for i in range(len(X) - 1)])
    proj_distances = np.array([distance_metric(_X[i], _X[i + 1]) for i in range(len(_X) - 1)])
    orig_distances += 1.e-13
    error = sum((orig_distances - proj_distances) ** 2 / orig_distances)
    error /= sum(orig_distances)
    return error


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
    if config.read:
        d1, d2 = dataloader.read_datasets(config.file1, config.file2)
    else:
        d1, d2 = dataloader.generate_data_with_labels(config.file1, config.type)

    x1, y1 = d1
    x2, y2 = d2

    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(x1)

    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=5, c=y1, cmap='Spectral')
    plt.title('Embedding of the training set by UMAP ' + str(config.file1), fontsize=24)
    plt.savefig('../graphs/training ' + str(config.file1) + ".png")
    plt.plot()
    sammon_train = sammon_error(x1, trans.embedding_)
    #### test

    test_embedding = trans.transform(x2)
    plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s=5, c=y2, cmap='Spectral')

    plt.title('Embedding of the test set by UMAP ' + str(config.file1), fontsize=24);
    plt.savefig('../graphs/testing ' + str(config.file1) + ".png")
    plt.plot()
    sammon_test = sammon_error(x2, test_embedding)

    #### noise
    noisy = x2 + np.random.random(x2.shape) * 3
    noise_embedding = trans.transform(noisy)
    plt.scatter(noise_embedding[:, 0], noise_embedding[:, 1], s=5, c=y2, cmap='Spectral')

    plt.title('Embedding of the test set by UMAP ' + str(config.file1), fontsize=24);
    plt.savefig('../graphs/noise ' + str(config.file1) + ".png")
    plt.plot()
    sammon_noise = sammon_error(noisy, noise_embedding)

    #### random garbage
    noisy = np.random.random(x2.shape) * 16
    noise_embedding = trans.transform(noisy)
    plt.scatter(noise_embedding[:, 0], noise_embedding[:, 1], s=5, c=y2, cmap='Spectral')

    plt.title('Embedding of the test set by UMAP ' + str(config.file1), fontsize=24);
    plt.savefig('../graphs/random ' + str(config.file1) + ".png")
    plt.plot()
    sammon_random = sammon_error(noisy, noise_embedding)

    report = {'train': [sammon_train, 'training ' + str(config.file1) + ".png"],
              'test': [sammon_test, 'testing ' + str(config.file1) + ".png"],
              'noise': [sammon_noise, 'noise ' + str(config.file1) + ".png"],
              'random': [sammon_random, 'random ' + str(config.file1) + ".png"],
              }
    pprint(report)
    reporter.save_report(config, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # datasets params
    parser.add_argument('--file1', type=str, default='two_diamonds', choices=['mnist', 'svhn', 'usps', 'iris', 'wine',
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
    parser.add_argument('--type', type=str, default='class',
                        choices=['class', 'cluster_removal', 'test_split'])
    parser.add_argument('--read', type=bool, default=False)
    # stats params
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
    # logging
    parser.add_argument('--data_type', type=str, default='Continuous')
    parser.add_argument('--log_name', type=str, default='umap')
    # test specific parameters
    parser.add_argument('--proportiontocut', type=float, default=0.01)

    config = parser.parse_args()
    print(config)
    main(config)
