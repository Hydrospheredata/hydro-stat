import os
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from statistical_report.statistical_feature_report import HeatMapData

TEST_SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class TestHeatMapData:

    @pytest.fixture
    def h1(self):
        h = HeatMapData("x", "y",
                        ['1', '2'], ['a', 'b'],
                        np.array([0, 1, 1]), np.array([1, 0, 1]))
        return h

    @pytest.fixture
    def adult_heatmaps(self):
        df = pd.read_csv(f"{TEST_SCRIPT_PATH}/resources/training_data.csv")
        column_pairs = list(combinations(df.columns, 2))

        hs = []
        for c1_name, c2_name in column_pairs:
            c1 = df[c1_name]
            c1_le = LabelEncoder()
            c1_encoded = c1_le.fit_transform(c1)

            c2 = df[c2_name]
            c2_le = LabelEncoder()
            c2_encoded = c2_le.fit_transform(c2)

            h = HeatMapData(c1_name, c2_name,
                            c1_le.classes_, c2_le.classes_,
                            c1_encoded, c2_encoded)
            hs.append(h)
        return hs

    def test_integrity(self, h1: HeatMapData):
        # Test that all values in a columns sums to 1
        x_sum = np.sum(h1.intensity, axis=0)
        assert all(x_sum == 1)

    def test_adult_heatmaps_integrity(self, adult_heatmaps: List[HeatMapData]):
        for h in adult_heatmaps:
            # Test that all values in a columns sums to 1
            print(h.x_title, h.y_title)
            assert np.allclose(np.sum(h.intensity, axis=0).min(), 1, atol=1e2)

    def test_no_nans(self, adult_heatmaps: List[HeatMapData]):
        # Assert that there should be non Nones in a heatmap intensity values
        for h in adult_heatmaps:
            assert not np.isnan(np.min(h.intensity))

    def test_h1_intensity(self, h1: HeatMapData):
        true_values = np.array([[0, 0.5], [1, 0.5]])
        assert np.all(h1.intensity == true_values)
