import numpy as np
import pytest

from app.statistical_feature_report import HeatMapData


class TestHeatMapData:

    @pytest.fixture
    def h1(self):
        h = HeatMapData("x", "y",
                        ['1', '2'], ['a', 'b'],
                        np.array([0, 1, 1]), np.array([1, 0, 1]))
        return h

    def test_integrity(self, h1: HeatMapData):
        # Test that all values in a columns sums to 1
        x_sum = np.sum(h1.intensity, axis=0)
        assert all(x_sum == 1)

    def test_h1_intensity(self, h1: HeatMapData):
        true_values = np.array([[0, 0.5], [1, 0.5]])
        assert np.all(h1.intensity == true_values)
