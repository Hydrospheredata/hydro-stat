import logging
import os
import random

import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import boto3

import datasets.mnist as mnist
from datasets import svhn, usps
from numpy.random import choice
import random


def choose_k(data, max):
    # add more algorithms
    choices = ['best', 'random']
    probabilities = [0.2, 0.8]
    draw = choice(choices, 1, p=probabilities)
    if draw == 'best':
        best_k = 2
        best_score = 0
        for k in range(2, max + 1):
            kmeans = KMeans(n_clusters=k)
            pred_cluster = kmeans.fit_predict(data)
            score = silhouette_score(data, pred_cluster)
            if score > best_score:
                best_k = k
                best_score = score
    else:
        return random.randint(2, max)
    return best_k


def _fake_dataset(d, type, classes=None):
    logger = logging.getLogger('fake_data_logger')
    if isinstance(d, tuple):
        x, y = d
    else:
        x, y = d[:, :-1], d[:, -1]

    if type == 'class':
        max = y.max()
        label = random.randint(0, max)
        d1 = x[y <= label]
        yd1 = y[y <= label]
        d2 = x[y >= label]
        yd2 = y[y >= label]
        d2 = np.array(d2, dtype=np.float64)
        d1 = np.array(d1, dtype=np.float64)
        logger.info('removed_class:{}, classes:{}, len_d1:{}, len_d2:{}'.format(label, str(classes), len(d1), len(d2)))
    elif type == 'cluster_removal':
        k = y.max()
        additional = np.random.randint(2, 12, 1)[0]
        k = choose_k(x, k + additional)
        cluster = random.randint(0, k - 1)
        kmeans = KMeans(n_clusters=k)
        pred_cluster = kmeans.fit_predict(x)
        d1 = x[pred_cluster <= cluster]
        yd1 = y[pred_cluster <= cluster]

        d2 = x[pred_cluster <= cluster]
        yd2 = y[pred_cluster <= cluster]
        logger.info(
            'cluster:{}, random_state:{}, len_d1:{}, len_d2:{}'.format(cluster, np.random.get_state()[1][0], len(d1),
                                                                       len(d2)))
        d2 = np.array(d2, dtype=np.float64)
        d1 = np.array(d1, dtype=np.float64)
    elif type == 'test_split':
        test_size = random.uniform(0.2, 0.5)
        random_state = random.randint(0, 5000)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        d1 = X_train
        d2 = X_test
        logger.info(
            'test_size:{}, random_state:{}, len_d1:{}, len_d2:{}'.format(test_size, random_state, len(d1),
                                                                         len(d2)))
    else:
        raise

    # if isinstance(d, tuple):
    #     if not isinstance(d1, tuple):
    #         d1 = (d1, yd1)
    #     if not isinstance(d2, tuple):
    #         d2 = (d2, yd2)
    return d1, d2


def _fake_dataset_with_labels(d, type, classes=None):
    logger = logging.getLogger('fake_data_logger')
    if isinstance(d, tuple):
        x, y = d
    else:
        x, y = d[:, :-1], d[:, -1]

    if type == 'class':
        max = y.max()
        label = random.randint(0, max)
        d1 = x[y != label]
        yd1 = y[y != label]
        d2 = x[y == label]
        yd2 = y[y == label]
        d2 = np.array(d2, dtype=np.float64), np.array(yd2, dtype=np.float64)
        d1 = np.array(d1, dtype=np.float64), np.array(yd1, dtype=np.float64)
        logger.info(
            'removed_class:{}, classes:{}, len_d1:{}, len_d2:{}'.format(label, str(classes), len(yd1), len(yd2)))
    elif type == 'cluster_removal':
        k = y.max()
        additional = np.random.randint(2, 12, 1)[0]
        k = choose_k(x, k + additional)
        cluster = random.randint(0, k - 1)
        kmeans = KMeans(n_clusters=k)
        pred_cluster = kmeans.fit_predict(x)
        d1 = x[pred_cluster != cluster]
        yd1 = y[pred_cluster != cluster]

        d2 = x[pred_cluster == cluster]
        yd2 = y[pred_cluster == cluster]
        logger.info(
            'cluster:{}, random_state:{}, len_d1:{}, len_d2:{}'.format(cluster, np.random.get_state()[1][0], len(yd1),
                                                                       len(yd2)))
        d2 = np.array(d2, dtype=np.float64), np.array(yd2, dtype=np.float64)
        d1 = np.array(d1, dtype=np.float64), np.array(yd1, dtype=np.float64)
    elif type == 'test_split':
        test_size = random.uniform(0.2, 0.5)
        random_state = random.randint(0, 5000)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        d1 = X_train, y_train
        d2 = X_test, y_test
        logger.info(
            'test_size:{}, random_state:{}, len_d1:{}, len_d2:{}'.format(test_size, random_state, len(X_train),
                                                                         len(X_test)))
    else:
        raise

    return d1, d2


def read_bike_sharing_london(file, target):
    data = pd.read_csv(file)
    data = data.drop(['timestamp'], axis=1)
    y = data[target].values
    X = data.drop([target], axis=1).values
    # X, y = data[:, 1:], data[:, 0]
    return X, y


def read_abalone(file, target):
    data = pd.read_csv(file, header=None)
    y = data.iloc[:, target].values
    X = data.drop([data.columns[target]], axis=1)
    if target != 0:
        X = data.drop([data.columns[0]], axis=1)
    X = X.values
    # print(X.max())
    # print(X.shape)
    # exit()
    return X, y


def read_auto(file, target):
    data = pd.read_csv(file, header=None, delim_whitespace=True)
    data = data.drop([data.columns[-1]], axis=1)
    data = data.drop([data.columns[3]], axis=1)
    y = data.iloc[:, target].values
    X = data.drop([data.columns[target]], axis=1)
    X = X.values
    return X, y


def read_absenteeism(file, target):
    data = pd.read_csv(file, sep=';')
    data = data.drop([data.columns[0]], axis=1)

    y = data.iloc[:, target].values
    X = data.drop([data.columns[target]], axis=1)
    X = X.values
    return X, y


def read_backnote(file):
    data = pd.read_csv(file, sep=',', header=None)
    target = -1
    y = data.iloc[:, target].values
    X = data.drop([data.columns[target]], axis=1)
    X = X.values
    return X, y


def read_mnist():
    mnist.init()
    x_train, y_train, x_test, y_test = mnist.load()
    X, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    X = X.reshape((-1, 28, 28))
    X = np.stack((X,) * 3, axis=-1)
    ranindex = random.randint(0, X.shape[0] - 1)
    plt.imshow(X[ranindex])
    plt.title('a random mnist image.')
    plt.show()
    return X, y


def read_svhn():
    svhn.init()
    x_train, y_train, x_test, y_test = svhn.load()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    X, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    ranindex = random.randint(0, X.shape[0] - 1)
    plt.imshow(X[ranindex])
    plt.title('a random svhn image.')
    plt.show()
    return X, y


def read_usps():
    usps.init()
    x_train, y_train, x_test, y_test = usps.load()
    X, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    X = X.reshape((-1, 28, 28))
    X = np.stack((X,) * 3, axis=-1)
    ranindex = random.randint(0, X.shape[0] - 1)
    plt.imshow(X[ranindex])
    plt.title('a random usps image.')
    plt.show()
    return X[:], y[:]


def read_wine(file):
    data = pd.read_csv(file).values
    X, y = data[:, 1:], data[:, 0]
    return X, y


def read_iris(file):
    data = pd.read_csv(file).values
    X, y = data[:, :-1], data[:, -1]
    return X, y


def clean_up(da):
    data = []
    for i in range(1, len(da)):
        data.append(float(da[i]))
    return data


def read_clustering_data(file):
    f = open(file + ".cls", "r")
    classes = {}
    for x in f:
        key, clas = x.split()
        if key != '%':
            classes[key] = clas
    f.close()
    data = {}
    f = open(file + ".lrn", "r")
    for x in f:
        sp = x.split()
        if sp[0] != '%':
            key, clas = sp[0], clean_up(sp)
            data[key] = clas
    f.close()
    X, y = np.array(list(data.values())), np.array(list(classes.values()))
    return X, y


## 33 DATASETS SO FAR
## 21


def read_file(file):
    if file == 'mnist':
        d = read_mnist()
    elif file == 'svhn':
        d = read_svhn()
    elif file == 'usps':
        d = read_usps()
    elif file == 'iris':
        d = read_iris("data/Continuous/iris.data")
    elif file == 'wine':
        d = read_wine("data/Continuous/wine.data")
    elif file == 'wing_nut':
        d = read_clustering_data("data/Continuous/WingNut")
    elif file == 'two_diamonds':
        d = read_clustering_data("data/Continuous/TwoDiamonds")
    elif file == 'tetra':
        d = read_clustering_data("data/Continuous/Tetra")
    elif file == 'target':
        d = read_clustering_data("data/Continuous/Target")
    elif file == 'lsun':
        d = read_clustering_data("data/Continuous/Lsun")
    elif file == 'Hepta':
        d = read_clustering_data("data/Continuous/Hepta")
    elif file == 'golf_ball':
        d = read_clustering_data("data/Continuous/GolfBall")
    elif file == 'engy_time':
        d = read_clustering_data("data/Continuous/EngyTime")
    elif file == 'chainlink':
        d = read_clustering_data("data/Continuous/Chainlink")
    elif file == 'atom':
        d = read_clustering_data("data/Continuous/Atom")
    elif file == 'bike_weather':
        d = read_bike_sharing_london('data/Continuous/london_merged.csv', 'weather_code')
    elif file == 'bike_holiday':
        d = read_bike_sharing_london('data/Continuous/london_merged.csv', 'is_holiday')
    elif file == 'bike_weekend':
        d = read_bike_sharing_london('data/Continuous/london_merged.csv', 'is_weekend')
    elif file == 'bike_season':
        d = read_bike_sharing_london('data/Continuous/london_merged.csv', 'season')
    elif file == 'abalone_rings':
        d = read_abalone('data/Continuous/abalone.csv', -1)
    elif file == 'abalone_sex':
        d = read_abalone('data/Continuous/abalone.csv', 0)
    elif file == 'auto_cylinders':
        d = read_auto('data/Continuous/auto-mpg.csv', 1)
    elif file == 'auto_origin':
        d = read_auto('data/Continuous/auto-mpg.csv', -1)
    elif file == 'absenteeism_reason':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 0)
    elif file == 'absenteeism_month':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 1)
    elif file == 'absenteeism_day':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 2)
    elif file == 'absenteeism_season':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 3)
    elif file == 'absenteeism_kids':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 12)
    elif file == 'absenteeism_alcohol':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 13)
    elif file == 'absenteeism_smoking':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 14)
    elif file == 'absenteeism_pet':
        d = read_absenteeism('data/Continuous/Absenteeism_at_work.csv', 15)
    elif file == 'backnote':
        d = read_backnote('data/Continuous/data_banknote_authentication.csv')
    # else:
    #     d = read_s3(file)
    else:
        d = pd.read_csv(file, encoding="ISO-8859-1").values
        d = (d, np.ones(d.shape[0]))
    return d


def read_datasets(file1, file2):
    d1, _ = read_file(file1)
    d2, _ = read_file(file2)
    return d1, d2


def read_datasets_un(file1, file2):
    print('here')
    x, y = read_file(file1)
    d1 = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    x, y = read_file(file2)
    d2 = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    print(d1.shape)
    print(d2.shape)
    return d1, d2


def generate_data(file1, type):
    x1, y1 = read_file(file1)
    lb_make = LabelEncoder()
    y1 = lb_make.fit_transform(y1)
    d1, d2 = _fake_dataset((x1, y1), type, lb_make.classes_)

    return d1, d2


def generate_data_with_labels(file1, type):
    x1, y1 = read_file(file1)
    lb_make = LabelEncoder()
    y1 = lb_make.fit_transform(y1)
    d1, d2 = _fake_dataset_with_labels((x1, y1), type, lb_make.classes_)

    return d1, d2


def read_iris(file):
    data = pd.read_csv(file).values
    X, y = data[:, :-1], data[:, -1]
    return X, y


def read_s3(file):
    file = 's3://hydro-drift-detection/london_merged.csv'
    bucketname = file.split('/')[2]  # replace with your bucket name
    filename = ''.join(file.split('/')[3:])  # replace with your object key
    saved = file.split('/')[-1]
    s3 = boto3.resource('s3')
    print(bucketname)
    print(filename)
    print(saved)
    s3.Bucket(bucketname).download_file(filename, saved)
    data = pd.read_csv(saved).values
    X, y = data[:, :-1], data[:, -1]
    if os.path.exists(saved):
        os.remove(saved)
    else:
        print("Can not delete the file as it doesn't exists")
    return X, y


if __name__ == '__main__':
    read_s3('s3://hydro-drift-detection/london_merged.csv')
