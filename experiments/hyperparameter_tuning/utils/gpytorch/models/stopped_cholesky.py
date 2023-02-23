import mlflow

from acgp.models.pytorch import estimators, ALL_POINTS, GPUStoppedCholesky, CPUStoppedCholesky
from hyperparameter_tuning.utils.abstract_hyper_parameter_tuning_algorithm import \
    AbstractHyperParameterTuningAlgorithm
from hyperparameter_tuning.utils.optimization_strategies.iterative_refinement import IterativeRefinement
from utils import registry
from utils.result_management.constants import BLOCK_SIZE, INITIAL_BLOCK_SIZE

ESTIMATOR = "ESTIMATOR"
MAX_N = "max_n"


class StoppedCholesky(AbstractHyperParameterTuningAlgorithm):
    @staticmethod
    def add_parameters_to_parser(parser):
        default_block_size = registry.default_block_size
        parser.add_argument("-bs", f"--{BLOCK_SIZE}", type=int, default=default_block_size) #np.iinfo(np.int64).max)
        parser.add_argument("-ibs", f"--{INITIAL_BLOCK_SIZE}", type=int, default=default_block_size)
        parser.add_argument("-e", "--" + ESTIMATOR, type=str, choices=estimators, default=ALL_POINTS)
        parser.add_argument("-mn", f"--{MAX_N}", type=int, default=50000)

    def __init__(self, X, y, k, sn2, mu, args, device="cpu", optimization_strategy=IterativeRefinement()):
        super().__init__(X, y, k, sn2, mu, args, device=device, optimization_strategy=optimization_strategy)
        error_tolerance = self.optimization_strategy().get_algorithm_tolerance
        self.estimator = args[ESTIMATOR]
        self.block_size = args[BLOCK_SIZE]
        # this parameter came later and for some runs it is not part of the saved argument dictionary
        if INITIAL_BLOCK_SIZE in args.keys():
            initial_block_size = args[INITIAL_BLOCK_SIZE]
        else:
            initial_block_size = self.block_size
        max_n = args[MAX_N]

        if device == "cuda":
            cholesky_class = GPUStoppedCholesky
        elif device == "cpu":
            cholesky_class = CPUStoppedCholesky
        else:
            raise RuntimeError("Unknown device %s" % str(device))
        self.stopped_cholesky = cholesky_class(X=X, y=y, k=k, sn2=sn2, mu=mu, estimator=self.estimator, max_n=max_n,
                                               error_tolerance=error_tolerance, block_size=self.block_size,
                                               initial_block_size=initial_block_size)

        self.require_ground_truth = initial_block_size < X.shape[0]

        self.set_tag(ESTIMATOR, self.estimator)
        self.set_tag(BLOCK_SIZE, self.block_size)
        self.set_tag(INITIAL_BLOCK_SIZE, initial_block_size)
        self.set_tag(MAX_N, max_n)

    def requires_ground_truth_recording(self):
        return self.require_ground_truth

    def get_name(self):
        return self.get_registry_key() + str(self.block_size)

    def log_metrics(self, step: int):
        mlflow.log_metric("FULLY_PROCESSED_DATAPOINTS", self.stopped_cholesky.last_iter, step=step)
        mlflow.log_metric("PARTIALLY_PROCESSED_POINTS", self.stopped_cholesky.last_advance, step=step)

    def create_loss_closure(self):
        return self.stopped_cholesky.create_loss_closure()

    def get_posterior(self, X_test, full_posterior=False):
        return self.stopped_cholesky.get_posterior(X_star=X_test, full_posterior=full_posterior)
