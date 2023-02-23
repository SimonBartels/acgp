import numpy as np
from time import thread_time
import mlflow

from acgp.hooks.abstract_hook import AbstractHook
from utils.result_management.constants import STEP_TIME
from utils.result_management.result_management import save_large_gpr_quantities


class RecordHook(AbstractHook):
    """
    Hook to record data that we can visualize our bounds for larger datasets.
    """
    def __init__(self, N, seed, preconditioner_size=0):
        """

        :param N: size of the Cholesky buffer -- NOT the dataset size
        :param seed: the seed we used to initialize the random number generator
        """
        super().__init__()
        self.seed = seed
        self.preconditioner_size = preconditioner_size
        self.diag = np.zeros(N)
        self.off_diag = np.zeros(N)
        self.t = np.zeros(N)
        self.time0 = None

    def prepare(self, *args):
        super().prepare(*args)
        self.time0 = thread_time()

    def pre_chol(self, idi: int, K_, y_):
        # stop timing for the copying
        mlflow.log_metric(STEP_TIME, thread_time() - self.time0, step=idi)

        advance = y_.shape[0]
        self.diag[idi:idi+advance] = np.diag(K_)
        self.off_diag[idi:idi+advance-1] = np.diag(K_, k=-1)
        self.t[idi:idi+advance] = y_[:, 0]

        # continue timing
        self.time0 = thread_time()
        return False

    def _finalize(self):
        mlflow.log_metric(STEP_TIME, thread_time() - self.time0, step=self.A.shape[0])
        save_large_gpr_quantities(self.seed, self.block_size, self.preconditioner_size, np.diag(self.A), self.y, self.diag, self.off_diag, self.t)
