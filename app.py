import copy
import json
import os
import sys
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from loguru import logger
from pandas.io.s3 import s3fs
from waitress import serve

from interpretability import profiler
from interpretability.interpret import interpret, get_types, get_cont, get_disc
from interpretability.monitor_stats import get_all, get_histograms
from metric_tests import continuous_stats
from metric_tests.discrete_stats import process_one_feature
import math
import random

import fastparquet
import pandas as pd
import requests
import s3fs
from hydrosdk.model import Model

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))

with open("version") as version_file:
    VERSION = version_file.read().strip()

THRESHOLD = 0.1

SUBSAMPLE_SIZE = 100
BATCH_SIZE = 10

HTTP_UI_ADDRESS = os.getenv("HTTP_UI_ADDRESS", "https://hydro-serving.dev.hydrosphere.io")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}
FEATURE_LAKE_BUCKET = "feature-lake"
BATCH_SIZE = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)

CORS(app)


def get_subsample(model: Model,
                  size: int,
                  s3fs: s3fs.S3FileSystem,
                  undersampling=False) -> pd.DataFrame:
    """
    Return a random subsample of request-response pairs from an S3 feature lake.


    :param undersampling: If True, returns subsample of size = min(available samples, size),
    if False and number of samples stored in feature lake < size then raise an Exception
    :param size: Number of requests\response pairs in subsample
    :type batch_size: Number of requests stored in each parquet file
    """

    number_of_parquets_needed = math.ceil(size / BATCH_SIZE)

    model_feature_store_path = f"s3://{FEATURE_LAKE_BUCKET}/{model.name}/{model.version}"

    parquets_paths = s3fs.find(model_feature_store_path)

    if len(parquets_paths) < number_of_parquets_needed:
        if not undersampling:
            raise ValueError(
                f"This model doesn't have {size} requests in feature lake.\n"
                f"Right now there are {len(parquets_paths) * BATCH_SIZE} requests stored.")
        else:
            number_of_parquets_needed = len(parquets_paths)

    selected_batch_paths = random.sample(parquets_paths, number_of_parquets_needed)

    input_fields_names = [field.name for field in model.contract.predict.inputs]
    output_fields_names = [field.name for field in model.contract.predict.outputs]
    columns_to_use = input_fields_names + output_fields_names

    dataset = fastparquet.ParquetFile(selected_batch_paths, open_with=s3fs.open)

    df: pd.DataFrame = dataset.to_pandas(columns=columns_to_use)

    if df.shape[0] > size:
        return df.sample(n=size)
    else:
        return df


def get_training_data(model: Model, fs: s3fs.S3FileSystem):
    s3_training_data_path = \
        requests.get(f"{HTTP_UI_ADDRESS}/monitoring/training_data?modelVersionId={model.id}").json()[0]
    training_data = pd.read_csv(fs.open(s3_training_data_path, mode='rb'))
    return training_data


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Domain Drift Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf8").strip()
    HEAD_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf8").strip()
    py_version = sys.version
    return jsonify({"version": VERSION,
                    "name": "domain-drift",
                    "pythonVersion": py_version,
                    "gitCurrentBranch": branch_name,
                    "available_routes": ["/buildinfo", "/", "/metrics"],
                    "gitHeadCommit": HEAD_COMMIT
                    })


def one_test(d1, d2, name):
    stats_type1, stats_type2 = tests_to_profiles.get(name, ['same', 'same'])
    s1 = profiler.get_statistic(d1, stats_type1, None, 1)
    s2 = profiler.get_statistic(d2, stats_type2, None, 2)
    report = continuous_stats.test(s1, s2, name, None)

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


@app.route("/metrics", methods=["GET"])
def get_metrics():
    possible_args = {"model_name", "model_version"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400
    try:
        model_name = request.args.get('model_name')
        model_version = int(request.args.get('model_version'))
    except:
        return jsonify({"message": f"Was unable to cast model_version to int"}), 400

    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova',
             'mann', 'kruskal', 'levene_mean',
             'levene_median', 'levene_trimmed',
             'sign_test', 'median_test',
             # 'min_max',
             'ks']
    logger.info("Metrics for model {} version {}".format(model_name, model_version))
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})

    cluster = Cluster(HTTP_UI_ADDRESS)
    model = Model.find(cluster, model_name, model_version)

    d2 = get_subsample(model, size=SUBSAMPLE_SIZE, s3fs=fs)
    d1 = get_training_data(model, fs)

    input_fields_names = [field.name for field in model.contract.predict.inputs]

    d1 = d1[input_fields_names]
    d2 = d2[input_fields_names]

    types1, types2 = get_types(d1), get_types(d2)
    d1 = d1.values
    d2 = d2.values
    (c1, cf1), (c2, cf2) = get_disc(d1, types1), get_disc(d2, types2)
    (d1, f1), (d2, f2) = get_cont(d1, types1), get_cont(d2, types2)
    stats1, stats2 = get_all(d1), get_all(d2)
    histograms = get_histograms(d1, d2, f1)

    full_report = {}
    pool = ThreadPool(processes=1)
    async_results = {}
    for test in tests:
        async_results[test] = pool.apply_async(one_test, (d1, d2, test))
    for test in tests:
        full_report[test] = async_results[test].get()
    per_statistic_change_probability(full_report)

    final_report = {'per_feature_report': fix(f1, stats1, histograms, stats2,
                                              per_statistic_change_probability(full_report),
                                              per_feature_change_probability(full_report)),
                    'overall_probability_drift': overall_probability_drift(full_report)}

    warnings = {'final_decision': final_decision(full_report),
                'report': interpret(final_report['per_feature_report'])}

    final_report['warnings'] = warnings

    if cf1 and len(cf1) > 0:
        for training_feature, deployement_feature, feature_name in zip(c1, c2, cf1):
            final_report['per_feature_report'][feature_name] = process_one_feature(training_feature,
                                                                                   deployement_feature)

    json_dump = json.dumps(final_report, cls=NumpyEncoder)
    return json.loads(json_dump)


@app.route("/config", methods=['GET', 'PUT'])
def get_params():
    if request.method == 'GET':
        return jsonify({'THRESHOLD': THRESHOLD})

    elif request.method == "PUT":
        possible_args = {"THRESHOLD"}
        if set(request.args.keys()) != possible_args:
            return jsonify(
                {"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

        logger.info('THRESHOLD changed from {} to {}'.format(THRESHOLD, request.args['THRESHOLD']))

        return Response(status=200)
    else:
        return Response(status=405)


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
