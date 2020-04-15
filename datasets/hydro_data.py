import math
import os
import random

import fastparquet
import pandas as pd
import requests
import s3fs
from hydrosdk.cluster import Cluster
from hydrosdk.model import Model

SUBSAMPLE_SIZE = 100
BATCH_SIZE = 10

FEATURE_LAKE_BUCKET = "feature-lake"
CLUSTER_URL = "https://hydro-serving.dev.hydrosphere.io"  # str(os.getenv("HTTP_UI_ADDRESS", "https://hydro-serving.dev.hydrosphere.io"))
MODEL_NAME = "adult_classification"
MODEL_VERSION = 26
S3_ENDPOINT = os.getenv("S3_ENDPOINT")


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


def get_deployment_data(model_name=MODEL_NAME, model_version=1):
    cluster = Cluster(CLUSTER_URL)
    model = Model.find(cluster, model_name, model_version)
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
    return get_subsample(model, size=SUBSAMPLE_SIZE, s3fs=fs)


def get_training_data(model_name=MODEL_NAME, model_version=26):
    s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT})
    cluster = Cluster(CLUSTER_URL)
    model = Model.find(cluster, model_name, model_version)

    s3_training_data_path = requests.get(f"{CLUSTER_URL}/training_data?modelVersionId={model.id}").json()[0]

    training_data = pd.read_csv(s3.open(s3_training_data_path, mode='rb'))

    return training_data
