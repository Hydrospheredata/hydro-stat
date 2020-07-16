import logging
from logging.config import fileConfig

import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion
from waitress import serve

from app.config import BUILD_INFO, DEBUG_ENV, HTTP_UI_ADDRESS, HTTP_PORT, \
    PRODUCTION_SUBSAMPLE_SIZE, SUPPORTED_DTYPES, BATCH_SIZE, S3_ENDPOINT
from app.statistical_report import StatisticalReport
from app.utils import get_training_data, get_production_data


fileConfig("resources/logging_config.ini")
hs_cluster = Cluster(HTTP_UI_ADDRESS)
app = Flask(__name__)
CORS(app)


@app.route("/stat/health", methods=['GET'])
def hello():
    return "Hi! I am Drift Report Service"


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


def is_model_supported(model_version: ModelVersion, subsample_size):
    has_training_data = len(requests.get(f"{HTTP_UI_ADDRESS}/monitoring/training_data?modelVersionId={model_version.id}").json()) > 0
    if not has_training_data:
        return False, "Need uploaded training data"

    production_data_aggregates = requests.get(f"{HTTP_UI_ADDRESS}/monitoring/checks/aggregates/{model_version.id}",
                                              params={"limit": 1, "offset": 0}).json()
    number_of_production_requests = production_data_aggregates['count']
    if number_of_production_requests == 0:
        return False, "Upload production data before analysing for data drift"
    elif number_of_production_requests * BATCH_SIZE < subsample_size:
        return False, f"Drift report is available after {subsample_size} requests." \
                      f" Currently ({number_of_production_requests * BATCH_SIZE}/{subsample_size})"

    signature = model_version.contract.predict

    input_tensor_shapes = [tuple(map(lambda dim: dim.size, input_tensor.shape.dim)) for input_tensor in signature.inputs]
    if not all([shape == tuple() for shape in input_tensor_shapes]):
        return False, "Data Drift is available only for signatures with all scalar fields"

    input_tensor_dtypes = [input_tensor.dtype for input_tensor in signature.inputs]
    if not all([dtype in SUPPORTED_DTYPES for dtype in input_tensor_dtypes]):
        return False, "Data Drift is available only for signatures with numerical and string fields"

    return True, "OK"


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
        # cluster = Cluster("https://hydro-serving.dev.hydrosphere.io/")
        model = ModelVersion.find_by_id(cluster, model_version_id)
    except Exception as e:
        logging.error(f"Failed to connect to the cluster {HTTP_UI_ADDRESS} or find the model there. {e}")
        return Response(status=500)

    try:
        logging.info(f"Loading training data. model version id = {model_version_id}")
        training_data = get_training_data(model, S3_ENDPOINT)
        # training_data = pd.read_csv("training_data.csv")
    except Exception as e:
        logging.error(f"Failed during loading training data. {e}")
        return Response(status=500)
    else:
        logging.info(f"Finished loading training data. model version id = {model_version_id}")

    try:
        logging.info(f"Loading production data. model version id = {model_version_id}")
        production_data = get_production_data(model, size=PRODUCTION_SUBSAMPLE_SIZE)
        # production_data = pd.read_csv("training_data.csv")

    except Exception as e:
        logging.error(f"Failed during loading production_data data. {e}")
        return Response(status=500)
    else:
        logging.info(f"Finished loading production data. model version id = {model_version_id}")

    try:
        report = StatisticalReport(model, training_data, production_data)
        report.process()
    except Exception as e:
        logging.error(f"Failed during computing statistics {e} {e.__traceback__}")
        return Response(status=500)

    return jsonify(report.to_json())


@app.route("/stat/config", methods=['GET', 'PUT'])
def get_params():
    global THRESHOLD
    if request.method == 'GET':
        return jsonify({'SIGNIFICANCE_LEVEL': THRESHOLD})

    elif request.method == "PUT":
        possible_args = {"SIGNIFICANCE_LEVEL"}
        if set(request.args.keys()) != possible_args:
            return jsonify(
                {"message": f"Expected args: {possible_args}. Provided args: {set(request.args.keys())}"}), 400

        logging.info('SIGNIFICANCE_LEVEL changed from {} to {}'.format(THRESHOLD, request.args['SIGNIFICANCE_LEVEL']))

        # TODO use mongo to store configs in there
        THRESHOLD = float(request.args['SIGNIFICANCE_LEVEL'])

        return Response(status=200)
    else:
        return Response(status=405)


if __name__ == "__main__":
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=HTTP_PORT)
    else:
        app.run(debug=True, host='0.0.0.0', port=HTTP_PORT)
