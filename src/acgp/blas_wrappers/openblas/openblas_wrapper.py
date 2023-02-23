import numpy as np
from ctypes import byref, c_int, c_char, c_double

from acgp.blas_wrappers.abstract_blas_wrapper import AbstractBlasWrapper
from acgp.blas_wrappers.openblas.blas_util import get_blas_object, c_double_p


Lchar = byref(c_char(b'L'[0]))
Rchar = byref(c_char(b'R'[0]))
Nchar = byref(c_char(b'N'[0]))
Tchar = byref(c_char(b'T'[0]))


class OpenBlasWrapper(AbstractBlasWrapper):
    def in_place_chol(self, A: np.ndarray):
        # if not np.isfortran(A):
        #     warnings.warn('Matrix is not FORTRAN contiguous!', RuntimeWarning)
        n = A.shape[0]
        lda = int(A.strides[1] / A.strides[0])
        A_p = A.ctypes.data_as(c_double_p)
        info = c_int(0)
        ret = get_blas_object().dpotrf_(Lchar, byref(c_int(n)), A_p, byref(c_int(lda)), byref(info))
        #info = info.value
        #check_info(info)

    def solve_triangular_inplace(self, L: np.ndarray, b: np.ndarray, transpose_b=True, lower=True, transpose_a=False,
                                 rside=True) -> ():
        assert(lower)  # lower=False is not implemented
        assert(rside)  # we can solve for vectors only from the right side--otherwise we run into memory alignmnent problems
        if not transpose_b:
            b = b.T

        transpose_a = transpose_a != rside

        # if transpose_b:
        if transpose_a:
            transa = Tchar  # we want to transpose
        else:
            transa = Nchar
        side = Rchar  # the matrix is on the RIGHT! of the equation system
        # else:
        #     transa = Nchar
        #     side = Lchar
        #     ldb = byref(c_int(int(b.strides[0] / b.strides[1])))
        ldb = byref(c_int(int(b.strides[1] / b.strides[0])))

        uplo = Lchar  # the Cholesky is lower
        diag = Nchar  # the matrix is not unit-diagonal
        m = byref(c_int(b.shape[0]))  # number of rows of b
        #m = byref(c_int(L.shape[0]))
        #n = byref(c_int(b.shape[1]))  # number of columns of b (equal to number of columns of A)
        n = byref(c_int(L.shape[0]))
        alpha = byref(c_double(1.))
        A_p = L.ctypes.data_as(c_double_p)
        #lda = n  # first dimension of A
        lda = byref(c_int(int(L.strides[1] / L.strides[0])))
        b_p = b.ctypes.data_as(c_double_p)
        #ldb = m  # first dimension of b
        get_blas_object().dtrsm_(side, uplo, transa, diag, m, n, alpha, A_p, lda, b_p, ldb)

    def symmetric_down_date(self, A, b):
        uplo = Lchar  # reference only lower part of A
        transa = Nchar  # we do not want to transpose b
        n = byref(c_int(A.shape[0]))
        k = byref(c_int(b.shape[1]))
        alpha = byref(c_double(-1.))
        b_p = b.ctypes.data_as(c_double_p)
        lda = byref(c_int(int(b.strides[1] / b.strides[0])))
        beta = byref(c_double(1.))
        A_p = A.ctypes.data_as(c_double_p)
        ldc = byref(c_int(int(A.strides[1] / A.strides[0])))
        get_blas_object().dsyrk_(uplo, transa, n, k, alpha, b_p, lda, beta, A_p, ldc)

    def dgemm(self, K, y, y_out):
        transa = Nchar
        transb = Nchar
        m = byref(c_int(K.shape[0]))
        k = byref(c_int(y.shape[0]))
        n = byref(c_int(y_out.shape[1]))
        alpha = byref(c_double(-1.))
        A_p = K.ctypes.data_as(c_double_p)
        lda = byref(c_int(int(K.strides[1] / K.strides[0])))
        b_p = y.ctypes.data_as(c_double_p)
        ldb = byref(c_int(int(y.shape[0])))
        beta = byref(c_double(1.))
        c_p = y_out.ctypes.data_as(c_double_p)
        ldc = byref(c_int(int(y_out.shape[0])))

        get_blas_object().dgemm_(transa, transb, m, n, k, alpha, A_p, lda, b_p, ldb, beta, c_p, ldc)
