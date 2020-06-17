import logging
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from statistical_feature_report import NumericalFeatureReport, CategoricalFeatureReport, StatisticalFeatureReport


class BivariateReportFactory:

    @classmethod
    def get_feature_report(cls, feature_report_1: StatisticalFeatureReport,
                           feature_report_2: StatisticalFeatureReport) -> 'BivariateFeatureReport':

        if isinstance(feature_report_1, NumericalFeatureReport):
            # Transform numerical data into categorical
            f1_labels, f1_training_data, f1_production_data = cls.__discretize_numerical_report(feature_report_1)
        elif isinstance(feature_report_1, CategoricalFeatureReport):
            f1_labels = feature_report_1.bins
            f1_training_data = feature_report_1.training_data
            f1_production_data = feature_report_1.production_data
        else:
            raise NotImplementedError(f"type {type(feature_report_1)} not supported")

        if isinstance(feature_report_2, NumericalFeatureReport):
            # Transform numerical data into categorical
            f2_labels, f2_training_data, f2_production_data = cls.__discretize_numerical_report(feature_report_2)
        elif isinstance(feature_report_2, CategoricalFeatureReport):
            f2_labels = feature_report_2.bins
            f2_training_data = feature_report_2.training_data
            f2_production_data = feature_report_2.production_data
        else:
            raise NotImplementedError(f"type {type(feature_report_2)} not supported")

        return BivariateFeatureReport(feature_report_1.feature_name, f1_labels, f1_training_data, f1_production_data,
                                      feature_report_2.feature_name, f2_labels, f2_training_data, f2_production_data)

    @staticmethod
    def __discretize_numerical_report(feature_report: StatisticalFeatureReport):
        bin_edges = feature_report.bins
        discretizer = KBinsDiscretizer()
        discretizer.set_params(bin_edges=bin_edges)

        labels = [f"{b1:.2f} <= {b2:.2f}" for b1, b2 in zip(bin_edges, bin_edges[1:])]

        return labels, discretizer.transform(feature_report.training_data), discretizer.transform(feature_report.production_data)


@dataclass
class DataGroup:
    name: str
    training_data: np.array
    production_data: np.array


@dataclass
class HeatMapData:
    x_title: str
    y_title: str
    x_labels: np.array
    y_labels: np.array
    x: np.array
    y: np.array

    intensity_list = []

    for y_label in y_labels:
        y_mask = y == y_label

        for x_label in x_labels:
            x_mask = x == x_label

            intensity = (x_mask & y_mask).sum()
            intensity_list.append(intensity)

    # HEATMAP OUTPUT
    intensity = np.array(intensity_list).reshape(len(y_labels) - 1, len(x_labels) - 1)

    def as_json(self):
        return {"x_axis_name": self.x_title,
                "y_axis_name": self.y_title,
                "x": self.x_labels.tolist(),
                "y": self.y_labels.tolist(),
                "density": self.intensity.tolist()}


@dataclass
class BivariateFeatureReport:
    logging.info(f"Creating bivariate report")

    f1_name: str
    f1_labels: np.array
    training_f1_labels: np.array
    production_f1_labels: np.array

    f2_name: str
    f2_labels: np.array
    training_f2_labels: np.array
    production_f2_labels: np.array

    # Calculate in self.process()
    production_heatmap: HeatMapData = None
    training_heatmap: HeatMapData = None

    # todo specify is ordinal or is categorical?!
    # if ordinal-ordinal, then KS-test is used
    # if categorical-categorical, then chisquare test is used
    drifted: bool = False

    def process(self):
        # TODO calculate GOF here?
        self.production_heatmap = HeatMapData(self.f1_name, self.f2_name,
                                              self.f1_labels, self.f2_labels,
                                              self.production_f1_labels, self.production_f2_labels)

        self.training_heatmap = HeatMapData(self.f1_name, self.f2_name,
                                            self.f1_labels, self.f2_labels,
                                            self.training_f1_labels, self.training_f2_labels)

    def as_json(self):
        # TOdO rename from and to ?
        # Drifted by groups?
        return {"from": self.f1_name,
                "to": self.f2_name,
                "drifted": self.drifted,
                "training_heatmap": self.training_heatmap.as_json(),
                "production_heatmap": self.production_heatmap.as_json()}

# # BoxPlot Report. Not used right now. Are there any cases we might prefer boxplot over heatmap?
# class NumericalBivariateReport(BivariateFeatureReport):
#     """
#     When target variable is numerical
#     """
#
#     def process(self, ):
#         self.training_plots: Dict = {}
#         self.production_plots: Dict = {}
#
#         for label in self.labels:
#             t_data = self.training_data[self.training_f1_labels == label]
#             p_data = self.production_data[self.production_f1_labels == label]
#             self.training_plots[label] = BoxPlotParameters(x=t_data)
#             self.production_plots[label] = BoxPlotParameters(x=p_data)
#
#
# @dataclass
# class BoxPlotParameters:
#     x: np.array
#     percentiles = np.percentile(x, [25, 50, 75])
#     q1 = percentiles[0]
#     median = percentiles[1]
#     q3 = percentiles[3]
#     iqr = q3 - q1
#     min = q1 - 1.5 * iqr
#     max = q3 + 1.5 * iqr
#
#     def to_json(self):
#         return {"q1": self.q1, "q3": self.q3, "median": self.median, "iqr": self.iqr, "min": self.min, "max": self.max}
