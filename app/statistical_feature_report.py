from typing import List, Optional, Dict

import numpy as np
from scipy import stats

from app.statistical_test import StatisticalTest


class StatisticalFeatureReport:

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        self.feature_name = feature_name
        self.training_data = training_data
        self.production_data = production_data
        self.tests = []
        self.is_processed = False
        self.drift_probability = None

    def process(self):
        pass

    def to_json(self):
        # TODO numpy encoder (?)
        pass

    def get_warning(self):
        pass


class NumericalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        super().__init__(feature_name, training_data, production_data)

        # Calculate bins to visualize histograms on UI
        bins, training_hist, deployment_hist = self._get_histogram()
        self.bins = bins
        self.training_histogram_values = training_hist
        self.production_histogram_values = deployment_hist

        # List of tests used for comparing production and training numerical columns
        self.tests: List[StatisticalTest] = [
            StatisticalTest("two_sample_t_test", np.mean, stats.ttest_ind, {"equal_var": False}),
            StatisticalTest("median_test", np.median, stats.median_test),
            StatisticalTest("levene", np.var, stats.levene, {"center": "mean"}),
        ]


    def process(self):
        for test in self.tests:
            test.process(self.training_data, self.production_data)

        # TODO add KS test to numerical features (or overall statistics)?
        # _, p_value = stats.ks_2samp(self.training_data, self.production_data)
        # self.ks_test_change = p_value <= 0.05

        self.is_processed = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])

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

    def _get_histogram(self, number_of_bins=20):
        training_data = self.training_data.astype(float)
        deployment_data = self.production_data.astype(float)

        data_minimum = min(training_data.min(), deployment_data.min())
        data_maximum = max(training_data.max(), deployment_data.max())

        training_histogram, bin_edges = np.histogram(training_data,
                                                     bins='auto',
                                                     normed=True, density=True,
                                                     range=[data_minimum, data_maximum])

        deployment_histogram, _ = np.histogram(deployment_data,
                                               bins=bin_edges,
                                               normed=True, density=True,
                                               range=[data_minimum, data_maximum])

        return bin_edges, training_histogram, deployment_histogram


class CategoricalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name, training_data, production_data):
        super().__init__(feature_name, training_data, production_data)
        # TODO calculate categories etc
