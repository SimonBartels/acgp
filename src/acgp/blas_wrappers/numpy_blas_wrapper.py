import numpy as np
from scipy.linalg import cho_factor, solve_triangular

from acgp.blas_wrappers.abstract_blas_wrapper import AbstractBlasWrapper


class NumpyBlasWrapper(AbstractBlasWrapper):
    def in_place_chol(self, K: np.ndarray):
        # BEWARE: cholesky and cho_factor behave differently!

        # TODO: somehow the overwrite does not work!
        # the problem seem to be views...
        #assert(not np.isfortran(K))
        #cho_factor(K, lower=lower, overwrite_a=True, check_finite=False)
        #return

        # the call below does NOT work with cholesky!
        L, _ = cho_factor(K, lower=True, overwrite_a=False, check_finite=False)
        K[:, :] = L

    def solve_triangular_inplace(self, L: np.ndarray, b: np.ndarray, transpose_b=True, transpose_a=False, lower=True) -> ():
        assert(lower)
        # if transpose_a:
        #     L = L.T
        #     lower = False
        if transpose_b:
            # TODO: make overwrite work!
            # the problem here seem to be views as well ...
            #solve_triangular(L, b.T, lower=lower, overwrite_b=True, check_finite=False)

            c = solve_triangular(L, b.T, lower=lower, overwrite_b=False, check_finite=False, trans=transpose_a)
            b[:, :] = c.T  # seems like the overwrite does not work
        else:
            #c = solve_triangular(L, b, lower=lower, overwrite_b=False, check_finite=False)
            #b[:, :] = c  # seems like the overwrite does not work
            c = solve_triangular(L, b, lower=lower, overwrite_b=False, check_finite=False, trans=transpose_a)
            b[:, :] = c

    def symmetric_down_date(self, A, b):
        A -= b @ b.T

    def dgemm(self, K, y, y_):
        y_ -= K @ y
