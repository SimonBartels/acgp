import mlflow

from hyperparameter_tuning.utils.optimization_strategies.default_strategy import DefaultOptimizationStrategy


class AbstractHyperParameterTuningAlgorithm:
    @staticmethod
    def add_parameters_to_parser(parser):
        raise NotImplementedError("abstract method")

    def create_loss_closure(self):
        raise NotImplementedError("abstract method")

    def get_posterior(self, X_test, full_posterior=False):
        """
        Computes posterior mean and uncertainty over the latent function.
        (That is, noise is NOT included!)
        :param X_test:
        :type X_test:
        :param full_posterior:
        :type full_posterior:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")

    def get_y_posterior(self, X_test, mu, fvar, full_posterior=False):
        if full_posterior:
            raise NotImplementedError("not supported")
        return mu, fvar + self.sn2()

    @classmethod
    def get_registry_key(cls):
        """
        Returns a key to identify the algorithm family.
        :return:
        """
        return cls.__name__

    def __init__(self, X, y, k, sn2, mu, args, device="cpu", optimization_strategy=DefaultOptimizationStrategy()):
        self.X = X
        self.y = y
        self.k = k
        self.sn2 = sn2
        self.mu = mu
        self.device = device
        self.opt_strategy = optimization_strategy
        # TODO: set tag for optimization strategy!

    def get_name(self):
        """
        Returns a name to identify that (may) distinguish algorithms from the same family when using different
        parameters.
        :return:
        """
        return self.get_registry_key()

    def optimization_strategy(self):
        return self.opt_strategy

    def log_metrics(self, step: int):
        pass

    def requires_ground_truth_recording(self):
        return True

    def get_named_tunable_parameters(self) -> [tuple]:
        return []

    def set_tag(self, key, value):
        """
        Wrapper around mlflow.set_tag which allows easy disabling.
        :param key:
        :param value:
        :return:
        """
        mlflow.set_tag(key, value)
