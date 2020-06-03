import os
import sys

import git
from hydro_serving_grpc import DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_DOUBLE, DT_FLOAT, DT_HALF, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_STRING

DEBUG_ENV = bool(os.getenv("DEBUG_ENV", True))
HTTP_PORT = int(os.getenv("HTTP_PORT", 5000))

BATCH_SIZE = 10
PRODUCTION_SUBSAMPLE_SIZE = 200

NUMERICAL_DTYPES = {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_DOUBLE, DT_FLOAT, DT_HALF, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}
SUPPORTED_DTYPES = NUMERICAL_DTYPES  # .union({DT_STRING, })

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
