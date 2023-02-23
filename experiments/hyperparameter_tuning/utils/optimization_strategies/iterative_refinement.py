from hyperparameter_tuning.utils.optimization_strategies.default_strategy import DefaultOptimizationStrategy


ITERATIVE_REFINEMENT_STRATEGY = "ITERATIVE_REFINEMENT_STRATEGY"


class IterativeRefinement(DefaultOptimizationStrategy):
    def __init__(self):
        super().__init__()
        self.failed_attempts = 0
        self.r = 2. / 3.
        self.algo_tol = self.r

    def get_tolerance(self, result, attempt):
        self.algo_tol = self.r ** (attempt - self.failed_attempts + 1)
        return self.algo_tol

    def abort(self, result, attempt):
        # convergence reaching relative reduction is considered a success
        if not result.success:
            self.failed_attempts += 1
        return self.failed_attempts >= 2

    def get_algorithm_tolerance(self):
        return self.algo_tol
