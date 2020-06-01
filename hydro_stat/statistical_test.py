from typing import Callable


class StatisticalTest:
    threshold = 0.05

    def __init__(self, name: str,
                 statistic_func: Callable,
                 statistic_test_func: Callable,
                 statistic_test_func_kwargs=None):
        """
        Parameters
        ----------
        name - Name of the statistical test used
        statistic_func -  Function which is used to calculate tested statistic from np.array
        statistic_test_func - Function which is used to calculate test statistic from 2 np.arrays
        statistic_test_func_kwargs - kwargs passed to statistic_test_func
        """

        self.name = name
        self.has_changed = None

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

        test_statistic, test_p = self.statistic_test_func(training_data, production_data, **self.statistic_test_func_kwargs)
        self.test_statistic = test_statistic
        self.test_p = test_p
        self.has_changed = test_p <= self.threshold

        # TODO Statistical test by itself has no change probability
        # FIXME Change UI from probability to boolean "changed at alpha = 0.01 - Yes/No"
        self.change_probability = 0

    def as_json(self):
        return {"change_probability": self.change_probability,
                "deployment": self.production_statistic,
                "training": self.training_statistic}
