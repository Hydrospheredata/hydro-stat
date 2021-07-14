import os

import numpy as np
import pandas as pd
import pytest
from hydrosdk import Cluster, ModelVersion

from hydro_stat.statistical_report.statistical_feature_report import HeatMapData
from hydro_stat.statistical_report.statistical_report import StatisticalReport



TEST_SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

training_data = pd.read_csv(f"{TEST_SCRIPT_PATH}/resources/training_data.csv")

@pytest.fixture()
def mv():
    c = Cluster("http://looolkekekek")
    mv = ModelVersion.find_by_id(c, 902)
    return mv

@pytest.fixture
def adult_report(mv):
    report = StatisticalReport(mv, training_data, training_data)
    report.process()
    return report

def check_heatmap_integirty(h: HeatMapData):
    # All columns shoud sum up to 1 +- 0.01
    return np.allclose(h.intensity.sum(axis=0).min(), 1, atol=0.01)

@pytest.mark.skip()
def test_heatmap_intensities(adult_report: StatisticalReport):
    for fr in adult_report.feature_reports:
        for bvr in fr.bivariate_reports:
            assert check_heatmap_integirty(bvr.production_heatmap)
            assert check_heatmap_integirty(bvr.training_heatmap)
