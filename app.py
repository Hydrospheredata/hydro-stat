import json
import os
import sys
from itertools import compress
from multiprocessing.pool import ThreadPool

import git
import numpy as np
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydro_serving_grpc import DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_DOUBLE, DT_FLOAT, DT_HALF, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_STRING
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model
from loguru import logger
from waitress import serve

from hydro_stat.discrete import process_feature
from hydro_stat.interpret import interpret
from hydro_stat.monitor_stats import get_numerical_statistics, get_histograms
from stat_analysis import one_test, per_statistic_change_probability, make_per_feature_report, per_feature_change_probability, \
    final_decision, \
    overall_probability_drift

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))
HTTP_PORT = int(os.getenv("HTTP_PORT", 5000))

NUMERICAL_DTYPES = {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_DOUBLE, DT_FLOAT, DT_HALF, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}
SUPPORTED_DTYPES = NUMERICAL_DTYPES.union({DT_STRING, })

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


hs_cluster = Cluster(HTTP_UI_ADDRESS)
app = Flask(__name__)
CORS(app)


@app.route("/stat/health", methods=['GET'])
def hello():
    return "Hi! I am Domain Drift Service"


@app.route("/stat/buildinfo", methods=['GET'])
def buildinfo():
    return jsonify(BUILD_INFO)


def is_model_supported(model_version: Model):
    has_training_data = len(requests.get(f"{HTTP_UI_ADDRESS}/monitoring/training_data?modelVersionId={model_version.id}").json()) > 0

    if not has_training_data:
        return False, "Need uploaded training data"

    signature = model_version.contract.predict

    input_tensor_shapes = [tuple(map(lambda dim: dim.size, input_tensor.shape.dim)) for input_tensor in signature.inputs]
    if not all([shape == tuple() for shape in input_tensor_shapes]):
        return False, "Only signatures with all scalar fields are supported"

    input_tensor_dtypes = [input_tensor.dtype for input_tensor in signature.inputs]
    if not all([dtype in SUPPORTED_DTYPES for dtype in input_tensor_dtypes]):
        return False, "Only signatures with all numerical fields are supported"

    return True, "OK"


@app.route("/stat/support", methods=['GET'])
def model_support():
    try:
        model_version_id = int(request.args.get('model_version_id'))
    except:
        return jsonify({"message": f"Unable to process 'model_version_id' argument"}), 400

    model_version = Model.find_by_id(hs_cluster, model_version_id)
    supported, description = is_model_supported(model_version)
    support_status = {"supported": supported, "description": description}
    return jsonify(support_status)


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
    input_fields_dtypes = [field.dtype for field in model.contract.predict.inputs]

    try:
        logger.info(f"Loading training data. model version id = {model_version_id}")
        training_data = get_training_data(model, S3_ENDPOINT)
        training_data = training_data[input_fields_names].values
        logger.info(f"Finished loading training data. model version id = {model_version_id}")
    except Exception as e:
        logger.error(f"Failed during loading training data. {e}")
        return Response(status=500)

    try:
        logger.info(f"Loading production data. model version id = {model_version_id}")
        production_data = get_production_data(model)
        production_data = production_data[input_fields_names].values
        logger.info(f"Finished loading production data. model version id = {model_version_id}")
    except Exception as e:
        logger.error(f"Failed during loading production_data data. {e}")
        return Response(status=500)

    try:
        # Calculate numerical statistics first
        numerical_fields = [field_dtype in NUMERICAL_DTYPES for field_dtype in input_fields_dtypes]
        numerical_fields_names = list(compress(input_fields_names, numerical_fields))
        numerical_training_data = training_data[:, numerical_fields]
        numerical_production_data = production_data[:, numerical_fields]

        numerical_training_statistics = get_numerical_statistics(numerical_training_data)
        numerical_production_statistics = get_numerical_statistics(numerical_production_data)

        histograms = get_histograms(numerical_training_data, numerical_production_data, numerical_fields_names)

        # Run async numerical stat test
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

    final_report = {'per_feature_report': make_per_feature_report(numerical_fields_names, numerical_training_statistics, histograms,
                                                                  numerical_production_statistics,
                                                                  per_statistic_change_probability(full_report),
                                                                  per_feature_change_probability(full_report)),
                    'overall_probability_drift': overall_probability_drift(full_report)}

    # Process discrete string fields
    string_fields = [field_dtype == DT_STRING for field_dtype in input_fields_dtypes]
    string_training_data = training_data[:, string_fields]
    string_production_data = production_data[:, string_fields]

    for training_feature, deployment_feature, feature_name in zip(string_training_data, string_production_data,
                                                                  list(compress(input_fields_names, string_fields))):
        final_report['per_feature_report'][feature_name] = process_feature(training_feature, deployment_feature)

    warnings = {'final_decision': final_decision(full_report),
                'report': interpret(final_report['per_feature_report'])}

    final_report['warnings'] = warnings

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
        serve(app, host='0.0.0.0', port=HTTP_PORT)
    else:
        app.run(debug=True, host='0.0.0.0', port=HTTP_PORT)
