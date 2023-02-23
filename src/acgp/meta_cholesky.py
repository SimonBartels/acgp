import warnings

from acgp.blas_wrappers.abstract_blas_wrapper import AbstractBlasWrapper
from acgp.hooks.abstract_hook import AbstractHook


class MetaCholesky:
    """
    This class is a generic and efficient python wrapper for a Cholesky decomposition.
    The efficiency of this implementation depends on the efficiency of the imported low_level_ops.
    """
    def __init__(self, block_size=256, initial_block_size=10240, blaswrapper: AbstractBlasWrapper = None):
        if block_size // 2 - block_size / 2 != 0:
            raise RuntimeError("The blocksize must be a multiple of 2!")
        self.parameters = {"block_size": block_size}
        self.block_size = block_size
        self.initial_block_size = initial_block_size
        if blaswrapper is None:
            warnings.warn("Going to use extremely slow numpy wrappers for low level BLAS operations!")
            from acgp.blas_wrappers.numpy_blas_wrapper import NumpyBlasWrapper
            blaswrapper = NumpyBlasWrapper()

        self.in_place_chol = blaswrapper.in_place_chol
        self.solve_triangular_inplace = blaswrapper.solve_triangular_inplace
        self.symmetric_down_date = blaswrapper.symmetric_down_date
        self.dgemm = blaswrapper.dgemm

    def get_signature(self):
        return type(self).__name__ + str(self.block_size)

    def run_configuration(self, A, err, kernel_evaluator=lambda *args: None, hook=AbstractHook()) -> ():
        """
        Estimates the log-marginal likelihood of a Gaussian process from a subset.
        :param A: allocated memory where the Cholesky can be stored into
        :param err: targets - mu(X)
        :param kernel_evaluator: function that writes kernel entries and noise into given indices
        :param hook: callback that can decide to stop the Cholesky if desired
        :return:
        """
        lower = True  # we compute a lower triangular Cholesky decomposition
        N = A.shape[0]  # matrix size -- can be different from the total dataset size!
        assert(A.shape[1] == N)
        assert(err.shape[0] == N and err.shape[1] == 1)

        block_size = self.initial_block_size
        if block_size > A.shape[0]:
            block_size = A.shape[0]

        if hook.prepare(A, err, self.block_size):
            return

        # first iteration of the Cholesky
        kernel_evaluator(A, 0, block_size, 0, block_size)
        K_ = A[:block_size, :block_size]
        y_ = err[:block_size, :]
        if hook.pre_chol(idi=0, K_=K_, y_=y_):
            return
        self.in_place_chol(K_)
        # first iteration for solving the linear equation system
        self.solve_triangular_inplace(K_, y_, transpose_b=False, lower=lower)

        if hook.post_chol(idi=0, K_=K_, y_=y_):
            return
        # main loop of the Cholesky
        for idi in range(block_size, N, self.block_size):
            # make sure we never go beyond the size of the matrix
            advance = min(self.block_size, N - idi)

            # solve block off-diagonal part of the Cholesky for the already computed Cholesky
            kernel_evaluator(A, idi, advance, 0, idi)
            to = A[idi:idi + advance, :idi]  # the next part that we are going to write to
            self.solve_triangular_inplace(A[:idi, :idi], to, transpose_b=True, lower=lower)  # O(i^2 * B)
            # apply symmetric down-date
            kernel_evaluator(A, idi, advance, idi, advance)
            K_ = A[idi: idi + advance, idi: idi + advance]
            # K_ -= to @ to.T
            self.symmetric_down_date(K_, to)
            # start solving the next part of the linear equation system for the quadratic form
            y_ = err[idi:idi + advance, :]
            #y_ -= to @ y[:idi, :]
            self.dgemm(to, err[:idi, :], y_)
            if hook.pre_chol(idi=idi, K_=K_, y_=y_):
                return

            # perform Cholesky of the down-dated part
            self.in_place_chol(K_)  # O(B^3)
            # finish solving the linear equation system
            self.solve_triangular_inplace(K_, y_, transpose_b=False, lower=lower)

            if hook.post_chol(idi=idi, K_=K_, y_=y_):
                return
        hook.finalize()
        return
