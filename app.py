import json
import os
from multiprocessing.pool import ThreadPool

import git
import numpy as np
import sys
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from loguru import logger
from waitress import serve

from hydro_stat.discrete import process_feature
from hydro_stat.interpret import interpret, get_types, get_cont, get_disc
from hydro_stat.monitor_stats import get_all, get_histograms
from stat_analysis import one_test, per_statistic_change_probability, fix, per_feature_change_probability, \
    final_decision, \
    overall_probability_drift

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))

with open("version") as version_file:
    VERSION = version_file.read().strip()
    repo = git.Repo(".")
    BUILD_INFO = {
        "version": VERSION,
        "name": "hydro-stat",
        "pythonVersion": sys.version,
        "gitCurrentBranch": repo.active_branch.name,
        "gitHeadCommit": repo.active_branch.commit.hexsha
    }

THRESHOLD = 0.01

HTTP_UI_ADDRESS = os.getenv("HTTP_UI_ADDRESS", "http://managerui")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")

from hydro_stat.utils import get_production_data, get_training_data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
CORS(app)


@app.route("/stat/health", methods=['GET'])
def hello():
    return "Hi! I am Domain Drift Service"


@app.route("/stat/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILD_INFO)


@app.route("/stat/metrics", methods=["GET"])
def get_metrics():
    possible_args = {"model_version_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    try:
        model_version_id = int(request.args.get('model_version_id'))
    except:
        return jsonify({"message": f"Was unable to cast model_version to int"}), 400

    tests = ['two_sample_t_test', 'one_sample_t_test', 'anova',
             'mann', 'kruskal', 'levene_mean',
             'levene_median', 'levene_trimmed',
             'sign_test', 'median_test',
             'ks']
    logger.info("Metrics for model version id = {}".format(model_version_id))

    try:
        cluster = Cluster(HTTP_UI_ADDRESS)
        model = Model.find_by_id(cluster, model_version_id)
    except Exception as e:
        logger.error(f"Failed to connect to the cluster {HTTP_UI_ADDRESS} or find the model there. {e}")

    input_fields_names = [field.name for field in model.contract.predict.inputs]
    try:
        logger.info(f"Loading training data. model version id = {model_version_id}")
        training_data = get_training_data(model, S3_ENDPOINT)
        training_data = training_data[input_fields_names]
        logger.info(f"Finished loading training data. model version id = {model_version_id}")
    except Exception as e:
        logger.error(f"Failed during loading training data. {e}")
        return Response(status=500)

    try:
        logger.info(f"Loading production data. model version id = {model_version_id}")
        production_data = get_production_data(model)
        production_data = production_data[input_fields_names]
        logger.info(f"Finished loading production data. model version id = {model_version_id}")
    except Exception as e:
        logger.error(f"Failed during loading production_data data. {e}")
        return Response(status=500)

    try:
        training_data_types, production_data_types = get_types(training_data), get_types(production_data)
        training_data, production_data = training_data.values, production_data.values
        (c1, cf1), (c2, cf2) = get_disc(training_data, training_data_types), get_disc(production_data, production_data_types)
        (training_data, f1), (production_data, f2) = get_cont(training_data, training_data_types), get_cont(production_data,
                                                                                                            production_data_types)
        training_statistics, production_statistics = get_all(training_data), get_all(production_data)
        histograms = get_histograms(training_data, production_data, f1)

        full_report = {}
        pool = ThreadPool(processes=1)
        async_results = {}
        for test in tests:
            async_results[test] = pool.apply_async(one_test, (training_data, production_data, test))
        for test in tests:
            full_report[test] = async_results[test].get()
        per_statistic_change_probability(full_report)
    except Exception as e:
        logger.error(f"Failed during computing statistics {e}")
        return Response(status=500)

    final_report = {'per_feature_report': fix(f1, training_statistics, histograms, production_statistics,
                                              per_statistic_change_probability(full_report),
                                              per_feature_change_probability(full_report)),
                    'overall_probability_drift': overall_probability_drift(full_report)}

    warnings = {'final_decision': final_decision(full_report),
                'report': interpret(final_report['per_feature_report'])}

    final_report['warnings'] = warnings

    if cf1 and len(cf1) > 0:
        for training_feature, deployment_feature, feature_name in zip(c1, c2, cf1):
            final_report['per_feature_report'][feature_name] = process_feature(training_feature,
                                                                               deployment_feature)

    json_dump = json.dumps(final_report, cls=NumpyEncoder)
    return json.loads(json_dump)


@app.route("/stat/config", methods=['GET', 'PUT'])
def get_params():
    global THRESHOLD
    if request.method == 'GET':
        return jsonify({'THRESHOLD': THRESHOLD})

    elif request.method == "PUT":
        possible_args = {"THRESHOLD"}
        if set(request.args.keys()) != possible_args:
            return jsonify(
                {"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

        logger.info('THRESHOLD changed from {} to {}'.format(THRESHOLD, request.args['THRESHOLD']))

        # TODO use mongo to store configs in there
        THRESHOLD = float(request.args['THRESHOLD'])

        return Response(status=200)
    else:
        return Response(status=405)


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
