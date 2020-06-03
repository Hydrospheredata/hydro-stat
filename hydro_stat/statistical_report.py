from typing import List

import numpy as np
import pandas as pd
from hydrosdk.modelversion import ModelVersion

from config import NUMERICAL_DTYPES
from hydro_stat.statistical_feature_report import NumericalFeatureReport, StatisticalFeatureReport


class StatisticalReport:

    def __init__(self, model: ModelVersion, training_data: pd.DataFrame, production_data: pd.DataFrame):
        self.__is_processed = False

        # Get field names and dtypes for conditional analysis
        input_fields_names = [field.name for field in model.contract.predict.inputs]
        input_fields_dtypes = [field.dtype for field in model.contract.predict.inputs]
        field_name_to_dtype = dict(zip(input_fields_names, input_fields_dtypes))

        # Drop columns with all NANs from production and training data
        training_data.dropna(axis=1, how="all", inplace=True)
        production_data.dropna(axis=1, how="all", inplace=True)

        # Select common input field names available both in model signature, training data and production data
        common_input_field_names = set(training_data.columns). \
            intersection(set(input_fields_names)). \
            intersection(set(production_data.columns))

        numerical_fields_names = [field_name for field_name in common_input_field_names if
                                  field_name_to_dtype[field_name] in NUMERICAL_DTYPES]

        numerical_feature_reports = [NumericalFeatureReport(f_name,
                                                            training_data[f_name].values,
                                                            production_data[f_name].values) for f_name in numerical_fields_names]

        # string_fields = [field_dtype == DT_STRING for field_dtype in input_fields_dtypes]
        # string_fields_names = list(compress(input_fields_names, string_fields))
        #
        # string_feature_reports = [CategoricalFeatureReport(f_name,
        #                                                    training_data[f_name].values,
        #                                                    production_data[f_name].values) for f_name in string_fields_names]

        self.feature_reports: List[StatisticalFeatureReport] = numerical_feature_reports  # + string_feature_reports

    def process(self):
        [feature_report.process() for feature_report in self.feature_reports]
        self.__is_processed = True

    def to_json(self):
        if not self.__is_processed:
            raise ValueError("Called before calculating report")

        return {"overall_probability_drift": self.__overall_drift(),
                "per_feature_report": self.__per_feature_report(),
                "warnings": self.__warnings_report()}

    def __per_feature_report(self):
        return dict([(feature_report.feature_name, feature_report.to_json()) for feature_report in self.feature_reports])

    def __warnings_report(self):
        feature_warnings = [feature_report.get_warning() for feature_report in self.feature_reports]
        return list(filter(None, feature_warnings))

    def __overall_drift(self):
        return np.mean([feature_report.drift_probability for feature_report in self.feature_reports])
