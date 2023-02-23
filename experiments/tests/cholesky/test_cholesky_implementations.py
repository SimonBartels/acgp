import unittest
import warnings
import numpy as np

from acgp.hooks.stop_hook import StopHook
from utils.registry import get_fast_meta_cholesky, ALGORITHM_DICT


class TestCholeskyImplementations(unittest.TestCase):
    def test_exact_choleskies(self):
        """
        Tests for all exact Cholesky decompositions in our repertoire that they deliver the exact log-determinant.
        """
        chols = list(ALGORITHM_DICT.values())
        try:
            chols.append(get_fast_meta_cholesky(block_size=5 * 64))
        except:
            warnings.warn("Could not load fast Cholesky!")
        self.assertGreater(len(chols), 0)
        n = 1299
        sn2 = 1e-6
        K = np.random.rand(n, n)
        K = K.dot(K.T) + sn2 * np.eye(n)  # make matrix s.p.d.
        y = np.random.randn(n, 1)  # this is necessary for the correct memory layout for the fortran wrappers
        L = np.linalg.cholesky(K)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        t = np.linalg.solve(L, y)
        quad = np.sum(np.square(t))
        neg_llh = log_det / 2 + quad / 2 + n * np.log(2 * np.pi) / 2

        for chol in chols:
            L = np.asfortranarray(K.copy())
            hook = StopHook(N=n, min_noise=sn2)
            #y_ = np.asfortranarray(np.zeros([1, n])).T
            #y_[:, 0] = y[:, 0]
            y_ = y.copy()
            chol.run_configuration(L, y_, hook=hook)
            neg_llh_, _ = hook.get_bounds()
            self.assertAlmostEqual(0, (neg_llh - neg_llh_) / neg_llh)


if __name__ == '__main__':
    TestCholeskyImplementations().test_exact_choleskies()
