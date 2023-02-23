from acgp.hooks.abstract_hook import AbstractHook
from acgp.backends.numpy_backend import NumpyBackend
from acgp.bound_computation import Bounds, ExperimentalBounds


class StopHook(AbstractHook):
    def __init__(self, N: int, min_noise: float, noise_func=None, absolute_tolerance=0., relative_tolerance=0., delta=1.,
                 backend=NumpyBackend()):
        """
        :param N: overall size of the dataset (can be different from the allocated memory!)
        :param absolute_tolerance: absolute tolerance on the approximation error to the log-marginal likelihood
        :param relative_tolerance: relative tolerance on the approximation error to the log-marginal likelihood
        :param delta: tolerated failure chance of the error
        """
        super().__init__()
        self.backend = backend
        self.N = N
        self.C = self.N * backend.log(2 * backend.pi) / 2  # constant part of the log-marginal likelihood
        self.bound_estimates = []  # list that will store the bound estimators in each iteration
        self.auxilary_variables = []
        # multiplication with np.ones(1) to make sure that this is a numpy array
        self.min_noise = min_noise * backend.ones(1)  # lowest possible variance that we might observe
        if noise_func is None:
            noise_func = lambda *args: self.min_noise
        self.noise_func = noise_func
        self.delta = delta
        # setup bounds estimator
        self.bounds_estimator = Bounds(delta=delta, N=N, min_noise=min_noise, backend=backend)

        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

        self.ldet = 0.  # initial log determinant
        self.quad = 0.  # initial quadratic form
        self.iteration = 0

    def pre_chol(self, idi: int, K_, y_):
        # Here's where we do the bounds check

        # compute bounds to each component of the log evidence
        advance = y_.shape[0]
        if advance <= 1:
            return False  # just continue
        Udet, Ldet, Uquad, Lquad, aux = self.bounds_estimator.get_bound_estimators_and_auxilary_quantities(
            t0=idi, log_sub_det=self.ldet, sub_quad=self.quad, A_diag=self.backend.diag(K_),
            A_diag_off=self.backend.diag(K_, k=-1), noise_diag=self.noise_func(idi+1, advance-1), y=y_)
        # do some result recording
        self.bound_estimates.append((Udet, Ldet, Uquad, Lquad))
        self.auxilary_variables.append(aux)

        # compute bounds to the log evidence
        max_neg_llh = Uquad / 2 + Udet / 2 + self.C
        min_neg_llh = Lquad / 2 + Ldet / 2 + self.C
        # check if we are sufficiently precise
        diff = abs(max_neg_llh - min_neg_llh)  # sometimes, especially when experimenting, the lower bound can be higher than the upper
        return diff < self.absolute_tolerance \
               or diff / min(abs(max_neg_llh), abs(min_neg_llh)) / 2 < self.relative_tolerance

    def post_chol(self, idi: int, K_, y_):
        # updates log determinant and quadratic form for the fully processed datapoints

        # update quantities for processed subset
        self.ldet += 2 * self.backend.sum(self.backend.log(self.backend.diag(K_)))
        self.quad += self.backend.sum(self.backend.square(y_))
        self.iteration = idi + y_.shape[0]
        return False

    def _finalize(self):
        self.bound_estimates.append((self.ldet, self.ldet, self.quad, self.quad))

    def get_bounds(self):
        """
        Returns the recorded quantities and an estimate for the log evidence.
        :return:
        estimator, recorded values
        """
        Udet, Ldet, Uquad, Lquad = self.bound_estimates[-1]
        max_neg_llh = Uquad / 2 + Udet / 2 + self.C
        min_neg_llh = Lquad / 2 + Ldet / 2 + self.C
        return max_neg_llh / 2 + min_neg_llh / 2, self.bound_estimates


class ExploratoryStopHook(StopHook):
    """
    Uses an exploratory bound computation instead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds_estimator = ExperimentalBounds(delta=self.delta, N=self.N, min_noise=self.min_noise)
