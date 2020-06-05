import json
from typing import List

import numpy as np
import pandas as pd
from hydrosdk.modelversion import ModelVersion

from statistical_feature_report import StatisticalFeatureReport, FeatureReportFactory
from utils import NumpyArrayEncoder


class StatisticalReport:

    def __init__(self, model: ModelVersion, training_data: pd.DataFrame, production_data: pd.DataFrame):
        self.__is_processed = False

        # Drop columns with all NANs from production and training data
        training_data.dropna(axis=1, how="all", inplace=True)
        production_data.dropna(axis=1, how="all", inplace=True)

        # Select common input field names available both in model signature, training data and production data
        input_fields_names = [field.name for field in model.contract.predict.inputs]
        common_input_field_names = set(training_data.columns). \
            intersection(set(input_fields_names)). \
            intersection(set(production_data.columns))
        common_input_fields = [field for field in model.contract.predict.inputs if field.name in common_input_field_names]

        feature_reports = [FeatureReportFactory.get_feature_report(field.name, field.dtype,
                                                                   training_data[field.name],
                                                                   production_data[field.name]) for field in common_input_fields]

        self.feature_reports: List[StatisticalFeatureReport] = [x for x in feature_reports if x is not None]

    def process(self):
        [feature_report.process() for feature_report in self.feature_reports]
        self.__is_processed = True

    def to_json(self):
        if not self.__is_processed:
            raise ValueError("Called before calculating report")

        numpy_json = {"overall_probability_drift": self.__overall_drift(),
                      "per_feature_report": self.__per_feature_report(),
                      "warnings": self.__warnings_report()}

        encoded_numpy_json = json.dumps(numpy_json, cls=NumpyArrayEncoder)  # use dump() to write array into file
        return json.loads(encoded_numpy_json)

    def __per_feature_report(self):
        return dict([(feature_report.feature_name, feature_report.to_json()) for feature_report in self.feature_reports])

    def __warnings_report(self):
        feature_warnings = [feature_report.get_warning() for feature_report in self.feature_reports]
        feature_warnings = list(filter(None, feature_warnings))
        return {"final_decision": None, "report": feature_warnings}

    def __overall_drift(self):
        return np.mean([feature_report.drift_probability for feature_report in self.feature_reports])
