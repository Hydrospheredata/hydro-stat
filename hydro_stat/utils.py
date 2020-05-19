from typing import Union

import pandas as pd
import requests
import s3fs

from hydrosdk.model import Model, ExternalModel


def get_training_data(model: Union[Model, ExternalModel], s3_endpoint) -> pd.DataFrame:
    # TODO change this when model and extModel id is named same
    if isinstance(model, Model):
        r = requests.get(f"{model.cluster.http_address}monitoring/training_data?modelVersionId={model.id}")
    elif isinstance(model, ExternalModel):
        r = requests.get(f"{model.cluster.http_address}monitoring/training_data?modelVersionId={model.id_}")

    if r.status_code != 200:
        raise ValueError("Unable to fetch training data")
    if not r.json():
        raise ValueError("Training data not found")

    s3_training_data_path = r.json()[0]

    if s3_endpoint:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint})
        training_data = pd.read_csv(fs.open(s3_training_data_path, mode='rb'))
    else:
        training_data = pd.read_csv(s3_training_data_path)
    return training_data


def get_production_data(model: Union[Model, ExternalModel], size=1000) -> pd.DataFrame:
    # TODO change this when model and extModel id is named same
    if isinstance(model, Model):
        r = requests.get(f'{model.cluster.http_address}monitoring/checks/subsample/{model.id}?size={size}')
    elif isinstance(model, ExternalModel):
        r = requests.get(f'{model.cluster.http_address}monitoring/checks/subsample/{model.id_}?size={size}')

    if r.status_code != 200:
        raise ValueError("Unable to fetch production data")

    return pd.DataFrame.from_dict(r.json())
