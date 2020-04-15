import os
import sys

from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datasets.hydro_data import get_deployment_data, get_training_data
from interpretability.interpret import interpret, get_types, get_cont, get_disc
from interpretability.monitor_stats import get_all, get_histograms
from multiprocessing.pool import ThreadPool
import numpy as np
from interpretability import profiler
from metric_tests import continuous_stats
import copy
import json
from loguru import logger

from waitress import serve

import warnings

from metric_tests.discrete_stats import process_one_feature




DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))

with open("version") as version_file:
    VERSION = version_file.read().strip()

with open("params.json") as params:
    data = json.load(params)
    THRESHOLD = data['THRESHOLD']

tests_to_profiles = {'one_sample_t_test': ('mean', 'same'), 'sign_test': ('median', 'same'),
                     'min_max': ('min_max', 'same'),
                     'hull': ('delaunay', 'same')}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)

CORS(app)


@app.route("/", methods=['GET'])
def hello():
    return "Hi! I am Domain Drift Service"


@app.route("/buildinfo", methods=['GET'])
def buildinfo():
    branch_name = "metric_evaluation"  # check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf8").strip()
    py_version = sys.version
    return jsonify({"version": VERSION,
                    "name": "domain-drift",
                    "pythonVersion": py_version,
                    "gitCurrentBranch": branch_name,
                    "available_routes": ["/buildinfo", "/", "/metrics"],
                    # "gitHeadCommit": HEAD_COMMIT
                    })


def one_test(d1, d2, name):
    # logger.info(name)
    stats_type1, stats_type2 = tests_to_profiles.get(name, ['same', 'same'])
    s1 = profiler.get_statistic(d1, stats_type1, None, 1)
    s2 = profiler.get_statistic(d2, stats_type2, None, 2)
    report = continuous_stats.test(s1, s2, name, None)

    return report


def final_decision(full_report):
    count_pos = 0
    count_neg = 0
    # plogger.info(full_report)
    for key, log in full_report.items():
        # plogger.info(log)
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
    # return full_report


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
                    # 'min_max',
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
    except Exception as e:
        return jsonify({"message": f"Was unable to cast model_version to int"}), 400

    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova',
             'mann', 'kruskal', 'levene_mean',
             'levene_median', 'levene_trimmed',
             'sign_test', 'median_test',
             # 'min_max',
             'ks']
    full_report = {}
    d2 = get_deployment_data(model_name, 1)
    d1 = get_training_data(model_name, 26)

    cluster = Cluster("https://hydro-serving.dev.hydrosphere.io/")
    model = Model.find(cluster, "adult_classification", 1)
    input_fields_names = [field.name for field in model.contract.predict.inputs]
    output_fields_names = [field.name for field in model.contract.predict.outputs]
    # logger.info(input_fields_names + output_fields_names)
    # logger.info(input_fields_names)
    d1 = d1[input_fields_names]
    d2 = d2[input_fields_names]

    types1, types2 = get_types(d1), get_types(d2)
    d1 = d1.values
    d2 = d2.values
    (c1, cf1), (c2, cf2) = get_disc(d1, types1), get_disc(d2, types2)
    (d1, f1), (d2, f2) = get_cont(d1, types1), get_cont(d2, types2)
    stats1, stats2 = get_all(d1), get_all(d2)
    historgrams = get_histograms(d1, d2, f1, f2)
    # logger.info(d1.shape)
    # logger.info(d2.shape)
    pool = ThreadPool(processes=1)
    async_results = {}
    for test in tests:
        async_results[test] = pool.apply_async(one_test, (d1, d2, test))
    for test in tests:
        full_report[test] = async_results[test].get()
    warnings = {}
    per_statistic_change_probability(full_report)

    # logger.info(overall_probability_drift(full_report))
    # logger.info(per_statistic_change_probability(full_report))
    # logger.info(per_feature_change_probability(full_report))
    final_report = {}

    final_report['per_feature_report'] = fix(f1, stats1, historgrams, stats2,
                                             per_statistic_change_probability(full_report),
                                             per_feature_change_probability(full_report))

    final_report['overall_probability_drift'] = overall_probability_drift(full_report)
    warnings['final_decision'] = final_decision(full_report)
    warnings['report'] = interpret(final_report['per_feature_report'])
    final_report['warnings'] = warnings
    if cf1 and len(cf1) > 0:
        for trc, depc, namec in zip(c1, c2, cf1):
            final_report['per_feature_report'][namec] = process_one_feature(trc, depc)
    json_dump = json.dumps(final_report, cls=NumpyEncoder)

    return json.loads(json_dump)


@app.route("/config", methods=['GET', 'PUT'])
def get_params():
    possible_args = {"THRESHOLD"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    if request.method == 'GET':
        return jsonify({'THRESHOLD': THRESHOLD})

    elif request.method == "PUT":
        logger.info('THRESHOLD changed from {} to {}'.format(THRESHOLD, request.args['THRESHOLD']))
        with open("params.json", 'w') as params:
            params_dict = {"THRESHOLD": request.args['THRESHOLD']}
            json.dump(params_dict, params)

        return Response(status=200)
    else:
        return Response(status=405)


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
