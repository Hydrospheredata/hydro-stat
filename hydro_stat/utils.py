import pandas as pd
import requests
import s3fs
from hydrosdk.model import Model


def get_training_data(model: Model, s3_endpoint) -> pd.DataFrame:
    s3_training_data_path = requests.get(f"{model.cluster.http_address}/monitoring/training_data?modelVersionId={model.id}").json()[0]

    if s3_endpoint:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint})
        training_data = pd.read_csv(fs.open(s3_training_data_path, mode='rb'))
    else:
        training_data = pd.read_csv(s3_training_data_path)
    return training_data


def get_production_data(model: Model, size=1000) -> pd.DataFrame:
    r = requests.get(f'{model.cluster.http_address}/monitoring/checks/subsample/{model.id}?size={size}')
    if r.status_code != 200:
        raise ValueError("Unable to fetch production data")
    return pd.DataFrame.from_dict(r.json())
