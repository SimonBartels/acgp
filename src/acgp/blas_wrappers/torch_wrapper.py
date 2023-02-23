import torch
from acgp.blas_wrappers.abstract_blas_wrapper import AbstractBlasWrapper


class TorchWrapper(AbstractBlasWrapper):
    def in_place_chol(self, K):
        with torch.no_grad():
            torch.linalg.cholesky(K, out=K)
        #K[:, :] = torch.linalg.cholesky(K)

    def solve_triangular_inplace(self, L, b, transpose_b=True, transpose_a=False, lower=True) -> ():
        with torch.no_grad():
            # TODO: WHY ON EARTH IS TORCH MAKING A COPY OF THE COEFFICIENT MATRIX?!?!?!
            out_tensor = torch.zeros_like(L)
            # if transpose_a:
            #     L = L.T
            assert(lower)
            if transpose_b:
                torch.triangular_solve(b.T, L, upper=False, out=(b.T, out_tensor), transpose=transpose_a)
                #torch.linalg.solve_triangular(L, b.T, upper=False, out=b.T)
                #c, _ = torch.triangular_solve(b.T, L, upper=False)
                #b[:] = c.T
            else:
                torch.triangular_solve(b, L, upper=False, out=(b, out_tensor), transpose=transpose_a)
                #torch.linalg.solve_triangular(L, b, upper=False, out=b)
                #c, _ = torch.triangular_solve(b, L, upper=False)
                #b[:] = c

    def symmetric_down_date(self, A, b):
        with torch.no_grad():
            A -= b @ b.T

    def dgemm(self, K, y, y_):
        with torch.no_grad():
            y_ -= K @ y
