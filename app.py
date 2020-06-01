import json
import logging
import os
import sys
from logging.config import fileConfig

import git
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydro_serving_grpc import DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_DOUBLE, DT_FLOAT, DT_HALF, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_STRING
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from waitress import serve

from hydro_stat.statistical_report import StatisticalReport

fileConfig("logging_config.ini")

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))
HTTP_PORT = int(os.getenv("HTTP_PORT", 5000))

PRODUCTION_SUBSAMPLE_SIZE = 200

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

from hydro_stat.utils import get_production_data, get_training_data, is_model_supported


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


@app.route("/stat/support", methods=['GET'])
def model_support():
    try:
        model_version_id = int(request.args.get('model_version_id'))
    except:
        return jsonify({"message": f"Unable to process 'model_version_id' argument"}), 400

    model_version = ModelVersion.find_by_id(hs_cluster, model_version_id)
    supported, description = is_model_supported(model_version, PRODUCTION_SUBSAMPLE_SIZE)
    support_status = {"supported": supported, "description": description}
    return jsonify(support_status)


@app.route("/stat/metrics", methods=["GET"])
def get_metrics():
    possible_args = {"model_version_id"}
    if set(request.args.keys()) != possible_args:
        return jsonify({"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

    try:
        model_version_id = int(request.args.get('model_version_id'))
    except ValueError:
        return jsonify({"message": f"Was unable to cast model_version to int"}), 400

    logging.info("Calculating metrics for model version id = {}".format(model_version_id))

    try:
        cluster = Cluster(HTTP_UI_ADDRESS)
        model = ModelVersion.find_by_id(cluster, model_version_id)
    except Exception as e:
        logging.error(f"Failed to connect to the cluster {HTTP_UI_ADDRESS} or find the model there. {e}")
        return Response(status=500)

    try:
        logging.info(f"Loading training data. model version id = {model_version_id}")
        training_data = get_training_data(model, S3_ENDPOINT)
        logging.info(f"Finished loading training data. model version id = {model_version_id}")
    except Exception as e:
        logging.error(f"Failed during loading training data. {e}")
        return Response(status=500)
    else:
        logging.info(f"Finished loading training data. model version id = {model_version_id}")

    try:
        logging.info(f"Loading production data. model version id = {model_version_id}")
        production_data = get_production_data(model, size=PRODUCTION_SUBSAMPLE_SIZE)
    except Exception as e:
        logging.error(f"Failed during loading production_data data. {e}")
        return Response(status=500)
    else:
        logging.info(f"Finished loading production data. model version id = {model_version_id}")

    try:
        report = StatisticalReport(model, training_data, production_data)
        report.process()
    except Exception as e:
        logging.error(f"Failed during computing statistics {e}")
        return Response(status=500)

    return jsonify(report.to_json())


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

        logging.info('THRESHOLD changed from {} to {}'.format(THRESHOLD, request.args['THRESHOLD']))

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
