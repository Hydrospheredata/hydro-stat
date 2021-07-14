import json
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from hydrosdk.modelversion import ModelVersion

from hydro_stat.statistical_report.statistical_feature_report import StatisticalFeatureReport, FeatureReportFactory
from hydro_stat.utils.utils import NumpyArrayEncoder


def common_fields(signature, training_columns, production_columns):
    field_names = [field.name for field in signature]
    common_field_names = set(training_columns). \
        intersection(set(field_names)). \
        intersection(set(production_columns))
    return [field for field in signature if field.name in common_field_names]


class StatisticalReport:

    def __init__(self, model: ModelVersion, training_data: pd.DataFrame, production_data: pd.DataFrame):
        self.__is_processed = False

        # Drop columns with all NANs from production and training data
        training_data.dropna(axis=1, how="all", inplace=True)
        production_data.dropna(axis=1, how="all", inplace=True)

        # Select common field names available both in model signature, training data and production data
        common_input_fields = common_fields(model.signature.inputs, training_data.columns, production_data.columns)
        common_output_fields = common_fields(model.signature.outputs, training_data.columns, production_data.columns)

        input_feature_reports = [FeatureReportFactory.get_feature_report(field.name, field.dtype,
                                                                         training_data[field.name],
                                                                         production_data[field.name]) for field in common_input_fields]

        output_feature_reports = [FeatureReportFactory.get_feature_report(field.name, field.dtype,
                                                                          training_data[field.name],
                                                                          production_data[field.name]) for field in common_output_fields]

        input_feature_reports = list(filter(None, input_feature_reports))
        output_feature_reports = list(filter(None, output_feature_reports))

        # Combine inputs and outputs to create bivariate reports inside input feature reports
        for inp_f, out_f in product(input_feature_reports, output_feature_reports):
            inp_f.combine(out_f)

        self.feature_reports: List[StatisticalFeatureReport] = input_feature_reports + output_feature_reports

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
