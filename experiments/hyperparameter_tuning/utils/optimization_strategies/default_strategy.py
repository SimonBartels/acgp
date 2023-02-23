OPTIMIZATION_STRATEGY = "OPTIMIZATION_STRATEGY"
DEFAULT_OPTIMIZATION_STRATEGY = "DEFAULT_OPTIMIZATION_STRATEGY"


class DefaultOptimizationStrategy:
    def __init__(self, tolerance=0., algorithm_tolerance=0.1):
        self.tolerance = tolerance
        self.algorithm_tolerance = algorithm_tolerance

    def get_tolerance(self, result, attempt):
        return self.tolerance

    def abort(self, result, attempt):
        return attempt >= 2

    def get_algorithm_tolerance(self):
        return self.algorithm_tolerance
