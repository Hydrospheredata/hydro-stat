from typing import Callable

from config import SIGNIFICANCE_LEVEL


class StatisticalTest:
    threshold = SIGNIFICANCE_LEVEL

    def __init__(self, name: str,
                 statistic_func: Callable,
                 statistic_test_func: Callable,
                 statistic_test_func_kwargs=None):
        """
        Parameters
        ----------
        name - Name of the statistical test used
        statistic_func -  Function which is used to calculate tested statistic from np.array
        statistic_test_func - Function which is used to calculate (test_statistic, test_p) from 2 np.arrays
        statistic_test_func_kwargs - kwargs passed to statistic_test_func
        """

        self.name = name
        self.has_changed = None
        self.message = None

        self.statistic_func = statistic_func
        self.training_statistic = None
        self.production_statistic = None

        self.statistic_test_func = statistic_test_func
        if statistic_test_func_kwargs:
            self.statistic_test_func_kwargs = statistic_test_func_kwargs
        else:
            self.statistic_test_func_kwargs = {}
        self.test_statistic = None
        self.test_p = None

    def process(self, training_data, production_data):
        self.training_statistic = self.statistic_func(training_data)
        self.production_statistic = self.statistic_func(production_data)

        try:
            test_statistic, test_p = self.statistic_test_func(training_data, production_data, **self.statistic_test_func_kwargs)[:2]
        except Exception as e:
            self.message = f"Unable to calculate statistic: {str(e)}"
            self.has_changed = False
        else:
            self.test_statistic = test_statistic
            self.test_p = test_p
            self.has_changed = test_p <= self.threshold

            if self.has_changed:
                self.message = f"Different at significance level = {self.threshold}"
            else:
                self.message = f"No significant difference at significance level = {self.threshold}"

    def as_json(self):
        return {"has_changed": bool(self.has_changed),
                "deployment": self.production_statistic,
                "training": self.training_statistic,
                "message": self.message}
