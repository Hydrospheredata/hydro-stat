import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

from config import NUMERICAL_DTYPES
from statistical_test import StatisticalTest
from test_messages import mean_test_message, median_test_message, variance_test_message, chi_square_message, unique_values_test_message


class FeatureReportFactory:

    @classmethod
    def get_feature_report(cls, feature_name: str, feature_dtype, training_data: pd.Series,
                           production_data: pd.Series) -> Optional['StatisticalFeatureReport']:

        unique_training_values = training_data.value_counts().shape[0]

        if unique_training_values / training_data.shape[0] <= 0.05 and unique_training_values <= 20:
            logging.info(f"{feature_name} is selected as Categorical")
            return CategoricalFeatureReport(feature_name, training_data.values.astype("str"), production_data.values.astype("str"))
        elif feature_dtype in NUMERICAL_DTYPES:
            logging.info(f"{feature_name} is selected as Numerical Continuous")
            return NumericalFeatureReport(feature_name, training_data.values, production_data.values)
        else:
            logging.info(f"{feature_name} is a non-categorical string, ignoring it")
            return None


class BivariateReportFactory:

    @classmethod
    def __encode_feature_report(cls, feature_report):
        if isinstance(feature_report, NumericalFeatureReport):
            # Transform numerical data into categorical
            labels, training_data, production_data = cls.__discretize_numerical_report(feature_report)
        elif isinstance(feature_report, CategoricalFeatureReport):
            labels, training_data, production_data = cls.__encode_categorical_report(feature_report)
        else:
            raise NotImplementedError(f"type {type(feature_report)} not supported")
        return labels, training_data, production_data

    @classmethod
    def get_feature_report(cls, feature_report_1,
                           feature_report_2):

        f1_labels, f1_training_data, f1_production_data = cls.__encode_feature_report(feature_report_1)
        f2_labels, f2_training_data, f2_production_data = cls.__encode_feature_report(feature_report_2)

        return BivariateFeatureReport(feature_report_1.feature_name, f1_labels, f1_training_data, f1_production_data,
                                      feature_report_2.feature_name, f2_labels, f2_training_data, f2_production_data)

    @staticmethod
    def __encode_categorical_report(feature_report: 'CategoricalFeatureReport'):
        """
        use OrdinalEncoder to encode categories into ints

        Parameters
        ----------
        feature_report

        Returns
        -------
        tuple:
        * List of labels used in human readable form e.g. ["Cat", "Dog", ..., "Frog"]
        * Ordinally encoded training data
        * Ordinally encoded production data
        """

        labels = feature_report.bins

        encoder = OrdinalEncoder()
        encoder.categories_ = np.array([labels])

        training_data = encoder.transform(feature_report.training_data.reshape(-1, 1)).flatten()
        production_data = encoder.transform(feature_report.production_data.reshape(-1, 1)).flatten()

        return labels, training_data, production_data

    @staticmethod
    def __discretize_numerical_report(feature_report):
        """
        Returns
        -------
        tuple:

        * List of labels used in human readable form e.g. ["<10", "10-15", ..., ">100"]
        * Ordinally encoded binned training data
        * Ordinally encoded binned production data
        """
        bin_edges = feature_report.bins

        discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
        discretizer.bin_edges_ = [np.array(bin_edges)]
        discretizer.n_bins_ = np.array([len(bin_edges) - 1])

        labels = np.array([f"{b1:.2f} <= {b2:.2f}" for b1, b2 in zip(bin_edges, bin_edges[1:])])

        return labels, discretizer.transform(feature_report.training_data.reshape(-1, 1)).flatten(), \
               discretizer.transform(feature_report.production_data.reshape(-1, 1)).flatten()


class StatisticalFeatureReport(ABC):

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        logging.info(f"Creating report for {feature_name}")
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

        self.bivariate_reports: List = []

    def process(self):
        logging.info(f"Calculating features for {self.feature_name}")
        for bv in self.bivariate_reports:
            bv.process()

    def to_json(self) -> Dict:
        return {"drift-probability": self.drift_probability,
                "histogram": {"bins": self.bins.tolist(),
                              "deployment": self.production_histogram_values.tolist(),
                              "training": self.training_histogram_values.tolist()},
                "statistics": dict([(test.name, test.as_json()) for test in self.tests]),
                "bivariate_reports": [r.as_json() for r in self.bivariate_reports]}

    def get_warning(self) -> Optional:
        if self.drift_probability > 0.75:
            changed_statistics = list(filter(None, [x.name for x in self.tests if x.has_changed]))
            return {"drift_probability_per_feature": self.drift_probability,
                    "message": f"The feature {self.feature_name} has drifted. Following statistics have changed: {changed_statistics}."}

    def combine(self, another: 'StatisticalFeatureReport'):
        """
        Combine two statistical feature reports from features x1 and x2 respectively to calculate conditional
        distribution X2|X1.

        Parameters
        ----------
        another StatisticalFeatureReport

        Returns
        -------
        """
        self.bivariate_reports.append(BivariateReportFactory.get_feature_report(self, another))

    @abstractmethod
    def _get_histogram(self) -> Tuple[np.array, np.array, np.array]:
        """

        Returns
        -------
        (bins, training PMF values, production PMF values)
        """
        pass


class NumericalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name: str, training_data: np.array, production_data: np.array):
        super().__init__(feature_name, training_data, production_data)

        # List of tests used for comparing production and training numerical columns
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Mean", np.mean, stats.ttest_ind, mean_test_message, {"equal_var": False}),
            StatisticalTest("Median", np.median, stats.median_test, median_test_message, {"ties": "ignore"}),
            StatisticalTest("Variance", np.var, stats.levene, variance_test_message, {"center": "mean"}),
        ]

    def process(self):
        super().process()
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
                                                     range=[data_minimum, data_maximum])

        deployment_histogram, _ = np.histogram(deployment_data,
                                               bins=bin_edges,
                                               range=[data_minimum, data_maximum])

        # Obtain PMF for binned features. np.hist returns PDF which could be less recognizable by non-data scientists
        training_histogram = training_histogram / training_histogram.sum()
        deployment_histogram = deployment_histogram / deployment_histogram.sum()

        return bin_edges, training_histogram, deployment_histogram


class CategoricalFeatureReport(StatisticalFeatureReport):

    def __init__(self, feature_name, training_data, production_data):
        super().__init__(feature_name, training_data, production_data)

        # List of tests used for comparing production and training categorical frequencies
        self.tests: List[StatisticalTest] = [
            StatisticalTest("Category densities",
                            lambda x: np.round(x, 3),
                            self.__chisquare,
                            chi_square_message),
            StatisticalTest("Unique Values",
                            lambda density: self.bins[np.nonzero(density)],
                            self.__unique_values_test,
                            unique_values_test_message),
        ]

    def __unique_values_test(self, training_density, production_density):
        # If we have categories with positive frequencies in production, but have no such categories in training
        if sum((production_density > 0) & (training_density == 0)) > 0:
            return None, 0  # Definitely Changed
        else:
            return None, 1  # Prob. not changed

    def __chisquare(self, training_density, production_density):
        production_sample_size = self.production_data.shape[0]
        # ChiSquare test compares Observed Frequencies to Expected Frequencies, so we need to change arguments placement
        return stats.chisquare(np.round(production_density * production_sample_size) + 1,
                               np.round(training_density * production_sample_size) + 1)

    def _get_histogram(self):
        training_categories, t_counts = np.unique(self.training_data, return_counts=True)
        production_categories, p_counts = np.unique(self.production_data, return_counts=True)

        # Calculate superset of categories
        common_categories = np.array(list(set(training_categories).union(set(production_categories))))

        production_category_to_count = dict(zip(production_categories, p_counts))
        training_category_to_count = dict(zip(training_categories, t_counts))

        # Calculate frequencies per category for training and production data
        training_counts_for_common_categories = np.array(
            [training_category_to_count.get(category, 0) for category in common_categories])
        production_counts_for_common_categories = np.array(
            [production_category_to_count.get(category, 0) for category in common_categories])

        # Normalise frequencies to density
        training_density = training_counts_for_common_categories / training_counts_for_common_categories.sum()
        production_density = production_counts_for_common_categories / production_counts_for_common_categories.sum()

        return common_categories, training_density, production_density

    def process(self):
        super().process()
        for test in self.tests:
            test.process(self.training_histogram_values, self.production_histogram_values)

        self.is_processed = True
        self.drift_probability = np.mean([test.has_changed for test in self.tests])


class HeatMapData:

    def __init__(self, x_title: str, y_title: str,
                 x_labels: np.array, y_labels: np.array,
                 x: np.array, y: np.array):
        """
        Container for heatmap data to plot on the UI. Calculates densities between ordinally encoded labels
        in x and y correspondingly
        Parameters
        ----------
        x_title x axis name
        y_title y axis name
        x_labels  list of human readable x labels
        y_labels  list of human readable y labels
        x Ordinaly encoded x
        y Ordinaly encoded y
        """
        self.x_title = x_title
        self.y_title = y_title
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.x = x
        self.y = y

        intensity_list = []

        # Computes heatmap density
        for ordinal_label_y, _ in enumerate(y_labels):
            y_mask = y == ordinal_label_y
            for ordinal_label_x, _ in enumerate(x_labels):
                x_mask = x == ordinal_label_x
                intensity = np.round(np.logical_and(x_mask, y_mask).mean(), 4)
                intensity_list.append(intensity)

        self.intensity = np.array(intensity_list).reshape(len(y_labels), len(x_labels))

    def as_json(self):
        return {"x_axis_name": self.x_title,
                "y_axis_name": self.y_title,
                "x": self.x_labels.tolist(),
                "y": self.y_labels.tolist(),
                "density": self.intensity.tolist()}


class BivariateFeatureReport:
    def __init__(self, f1_name: str, f1_labels: List[str], training_f1_labels: np.array, production_f1_labels: np.array,
                 f2_name: str, f2_labels: List[str], training_f2_labels: np.array, production_f2_labels: np.array):
        """

        Parameters
        ----------
        f1_name Name of a first feature
        f1_labels List of human-readable labels for constructing a heatmap
        training_f1_labels Ordinally encoded array of training labels
        production_f1_labels Ordinally encoded array of production labels
        f2_name
        f2_labels
        training_f2_labels
        production_f2_labels
        """
        logging.info(f"Creating bivariate report between {f1_name} and {f2_name}")

        self.f1_name = f1_name
        self.f1_labels = f1_labels
        self.training_f1_labels = training_f1_labels
        self.production_f1_labels = production_f1_labels

        self.f2_name = f2_name
        self.f2_labels = f2_labels
        self.training_f2_labels = training_f2_labels
        self.production_f2_labels = production_f2_labels

        # Calculate in self.process()
        self.production_heatmap: HeatMapData = None
        self.training_heatmap: HeatMapData = None

        # Todo specify is ordinal or is categorical?!
        # if ordinal-ordinal, then KS-test is used
        # if categorical-categorical, then chisquare test is used
        self.drifted: bool = False

    def process(self):
        # TODO calculate GOF here?
        self.production_heatmap = HeatMapData(x_title=self.f1_name, y_title=self.f2_name,
                                              x_labels=self.f1_labels, y_labels=self.f2_labels,
                                              x=self.production_f1_labels, y=self.production_f2_labels)

        self.training_heatmap = HeatMapData(x_title=self.f1_name, y_title=self.f2_name,
                                            x_labels=self.f1_labels, y_labels=self.f2_labels,
                                            x=self.training_f1_labels, y=self.training_f2_labels)

    def as_json(self):
        return {"feature_1": self.f1_name,
                "feature_2": self.f2_name,
                "drifted": self.drifted,
                "training_heatmap": self.training_heatmap.as_json(),
                "production_heatmap": self.production_heatmap.as_json()}
