import math
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
CLUSTER_URL = "https://hydro-serving.dev.hydrosphere.io"
MODEL_NAME = "adult_classification"
MODEL_VERSION = 26

fs = s3fs.S3FileSystem()


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
    return get_subsample(model, size=SUBSAMPLE_SIZE, s3fs=fs)


def get_training_data(model_name=MODEL_NAME, model_version=26):
    cluster = Cluster(CLUSTER_URL)
    model = Model.find(cluster, model_name, model_version)
    r = requests.get(f'https://hydro-serving.dev.hydrosphere.io/monitoring/training_data?modelVersionId={model.id}')
    return pd.read_csv(r.json()[0])
