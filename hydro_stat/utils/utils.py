from json import JSONEncoder
import numpy as np
import pandas as pd
import logging
import requests
import s3fs
from hydrosdk.modelversion import ModelVersion


class HealthEndpointFilter(logging.Filter):
    def filter(self, record):
        return not "/stat/health" in record.getMessage()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


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

    checksWithoutError = list(filter(lambda x: x.get('_hs_error') == None, r_json))

    return pd.DataFrame.from_dict(checksWithoutError)
