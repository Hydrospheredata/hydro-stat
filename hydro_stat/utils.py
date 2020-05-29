import pandas as pd
import requests
import s3fs
from hydrosdk.modelversion import ModelVersion

from app import HTTP_UI_ADDRESS, SUPPORTED_DTYPES


def is_model_supported(model_version: ModelVersion, subsample_size):
    has_training_data = len(requests.get(f"{HTTP_UI_ADDRESS}/monitoring/training_data?modelVersionId={model_version.id}").json()) > 0
    if not has_training_data:
        return False, "Need uploaded training data"

    production_data_aggregates = requests.get(f"{HTTP_UI_ADDRESS}/monitoring/checks/aggregates/{model_version.id}",
                                              params={"limit": 1, "offset": 0}).json()
    number_of_production_requests = production_data_aggregates['count']
    if number_of_production_requests == 0:
        return False, "Upload production data before running hydro-stat"
    elif number_of_production_requests < subsample_size:
        return False, f"hydro-stat is available after {subsample_size} requests." \
                      f" Currently ({number_of_production_requests}/{subsample_size})"

    signature = model_version.contract.predict

    input_tensor_shapes = [tuple(map(lambda dim: dim.size, input_tensor.shape.dim)) for input_tensor in signature.inputs]
    if not all([shape == tuple() for shape in input_tensor_shapes]):
        return False, "Only signatures with all scalar fields are supported"

    input_tensor_dtypes = [input_tensor.dtype for input_tensor in signature.inputs]
    if not all([dtype in SUPPORTED_DTYPES for dtype in input_tensor_dtypes]):
        return False, "Only signatures with numerical or string fields are supported"

    return True, "OK"


def get_training_data(model: ModelVersion, s3_endpoint: str) -> pd.DataFrame:
    r = requests.get(f"{model.cluster.http_address}/monitoring/training_data?modelVersionId={model.id}")

    if r.status_code != 200:
        raise ValueError(f"Unable to fetch training data. Status Code {r.status_code}, {str(r)}")
    r_json = r.json()
    if not r_json:
        raise ValueError("Training path is not in a request body")
    s3_training_data_path = r_json[0]

    if s3_endpoint:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint})
        training_data = pd.read_csv(fs.open(s3_training_data_path, mode='rb'))
    else:
        training_data = pd.read_csv(s3_training_data_path)

    return training_data


def get_production_data(model: ModelVersion, size=1000) -> pd.DataFrame:
    r = requests.get(f'{model.cluster.http_address}/monitoring/checks/subsample/{model.id}?size={size}')

    if r.status_code != 200:
        raise ValueError(f"Unable to fetch production data. Status Code {r.status_code}, {str(r)}")
    r_json = r.json()

    if not r_json:
        raise ValueError("Production data not found in request body")

    return pd.DataFrame.from_dict(r_json)
