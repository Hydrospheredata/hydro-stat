import os

import numpy as np
import pandas as pd
import pytest
from hydrosdk import Cluster, ModelVersion

from statistical_report.statistical_feature_report import HeatMapData
from statistical_report.statistical_report import StatisticalReport

c = Cluster("https://hydro-serving.dev.hydrosphere.io/")
mv = ModelVersion.find_by_id(c, 902)

TEST_SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

training_data = pd.read_csv(f"{TEST_SCRIPT_PATH}/resources/training_data.csv")


class TestHeatMapData:

    @pytest.fixture
    def adult_report(self):
        report = StatisticalReport(mv, training_data, training_data)
        report.process()
        return report

    def check_heatmap_integirty(self, h: HeatMapData):
        # All columns should sum up to 1 +- 0.01
        return np.allclose(h.intensity.sum(axis=0).min(), 1, atol=0.01)

    def test_heatmap_intensities(self, adult_report: StatisticalReport):
        for fr in adult_report.feature_reports:
            for bvr in fr.bivariate_reports:
                assert self.check_heatmap_integirty(bvr.production_heatmap)
                assert self.check_heatmap_integirty(bvr.training_heatmap)
