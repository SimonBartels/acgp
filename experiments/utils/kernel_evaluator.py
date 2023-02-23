from acgp.backends.numpy_backend import NumpyBackend


def get_kernel_evaluator(X, k, sn2, backend=NumpyBackend()):
    def kernel_evaluator(A, i0, i1, j0, j1):
        if i0 == j0 and i1 == j1:
            A[i0:i0 + i1, j0:j0 + j1] = k(X[i0:i0+i1, :])
            B = A[i0:i0+i1, j0:j0+j1]
            #backend.fill_diagonal(B, backend.diag(B) + sn2)
            B += sn2 * backend.eye(i1)
        elif j1 <= i0:
            A[i0:i0+i1, j0:j0+j1] = k(X[i0:i0+i1, :], X[j0:j0+j1, :])
        else:
            raise RuntimeError("This case should not occur")
    return kernel_evaluator
