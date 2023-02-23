import numpy as np


class AbstractBlasWrapper():
    def in_place_chol(self, K: np.ndarray):
        """
        Performs a Choleksy decomposition of K in K.
        :param K:
        :return:
        """
        raise NotImplementedError("abstract method")

    def solve_triangular_inplace(self, L: np.ndarray, b: np.ndarray, transpose_b=True, lower=True) -> ():
        """
        Solves a linear equation system with triangular coefficient matrix and overwrites the solution vector.
        :param L:
        :param b:
        :param transpose_b:
        :param lower:
        :return:
        """
        raise NotImplementedError("abstract method")

    def symmetric_down_date(self, A, b):
        """
        Performs A -= b @ b.T in place.
        :param A:
        :param b:
        :return:
        """
        raise NotImplementedError("abstract method")

    def dgemm(self, K, y, y_):
        """
        Performs y_ -= K @ y
        :param K:
        :param y:
        :param y_:
        :return:
        """
        raise NotImplementedError("abstract method")
