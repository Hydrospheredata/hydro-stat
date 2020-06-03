from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from scipy import stats

from config import NUMERICAL_DTYPES
from statistical_test import StatisticalTest


class FeatureReportFactory:

    @classmethod
    def get_feature_report(cls, feature_name: str, feature_dtype, training_data: pd.Series,
                           production_data: pd.Series) -> Optional['StatisticalFeatureReport']:

        unique_training_values = training_data.value_counts().shape[0]

        if unique_training_values / training_data.shape[0] <= 0.05:
            return CategoricalFeatureReport(feature_name, training_data.values.astype("str"), production_data.values.astype("str"))
        elif feature_dtype in NUMERICAL_DTYPES:
            return NumericalFeatureReport(feature_name, training_data.values, production_data.values)
        else:
            return None


class StatisticalFeatureReport(ABC):

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        self.feature_name = feature_name
        self.training_data = training_data
        self.production_data = production_data
        self.tests = []
        self.is_processed = False
        self.drift_probability = None

        # Calculate bins to visualize histograms on UI
        bins, training_hist, deployment_hist = self._get_histogram()
        self.bins = bins
        self.training_histogram_values = training_hist
        self.production_histogram_values = deployment_hist

    @abstractmethod
    def process(self):
        pass

    def to_json(self) -> Dict:
        return {"drift-probability": self.drift_probability,
                "histogram": {"bins": self.bins.tolist(),
                              "deployment": self.production_histogram_values.tolist(),
                              "training": self.training_histogram_values.tolist()},
                "statistics": dict([(test.name, test.as_json()) for test in self.tests])}

    def get_warning(self) -> Optional:
        if self.drift_probability > 0.75:
            return {"drift_probability_per_feature": self.drift_probability,
                    "message": f"The feature {self.feature_name} has changed."}

    @abstractmethod
    def _get_histogram(self):
        pass


class NumericalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        super().__init__(feature_name, training_data, production_data)

        # List of tests used for comparing production and training numerical columns
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Mean", np.mean, stats.ttest_ind, {"equal_var": False}),
            StatisticalTest("Median", np.median, stats.median_test, {"ties": "ignore"}),
            StatisticalTest("Variance", np.var, stats.levene, {"center": "mean"}),
        ]

    def process(self):
        for test in self.tests:
            test.process(self.training_data, self.production_data)

        # TODO add KS test to numerical features?
        # _, p_value = stats.ks_2samp(self.training_data, self.production_data)
        # self.ks_test_change = p_value <= 0.05

        self.is_processed = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])

    def _get_histogram(self):
        training_data = self.training_data.astype(float)
        deployment_data = self.production_data.astype(float)

        data_minimum = min(training_data.min(), deployment_data.min())
        data_maximum = max(training_data.max(), deployment_data.max())

        training_histogram, bin_edges = np.histogram(training_data,
                                                     bins='fd',
                                                     density=True,
                                                     range=[data_minimum, data_maximum])

        deployment_histogram, _ = np.histogram(deployment_data,
                                               bins=bin_edges,
                                               density=True,
                                               range=[data_minimum, data_maximum])

        return bin_edges, training_histogram, deployment_histogram


class CategoricalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name, training_data, production_data):
        super().__init__(feature_name, training_data, production_data)

        # List of tests used for comparing production and training categorical frequencies
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Chi-Squared", np.mean, stats.chisquare, ),
            StatisticalTest("Unique Values", np.nonzero, CategoricalFeatureReport.__unique_values_test),
        ]

    @classmethod
    def __unique_values_test(cls, training_frequencies, production_frequencies):
        if sum(training_frequencies[production_frequencies > 0]) > 0:
            return None, 0  # Definitely Changed
        else:
            return None, 1  # Prob. Not changed

    def _get_histogram(self):
        training_categories, t_counts = np.unique(self.training_data, return_counts=True)
        production_categories, p_counts = np.unique(self.production_data, return_counts=True)

        common_categories = list(set(training_categories).union(set(production_categories)))
        production_category_to_count = dict(zip(production_categories, p_counts))
        training_category_to_count = dict(zip(training_categories, t_counts))

        training_counts_for_common_categories = np.array([training_category_to_count.get(category, 0) for category in common_categories])
        production_counts_for_common_categories = np.array(
            [production_category_to_count.get(category, 0) for category in common_categories])

        # training_density = training_counts_for_common_categories / training_counts_for_common_categories.sum()
        # production_density = production_counts_for_common_categories / production_counts_for_common_categories.sum()

        return common_categories, training_counts_for_common_categories, production_counts_for_common_categories

    def process(self):
        for test in self.tests:
            test.process(self.training_histogram_values, self.production_histogram_values)

        self.is_processed = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])
