class AbstractHook:
    def __init__(self):
        self.A = None
        self.y = None
        self.block_size = None
        self.finished = False

    def pre_chol(self, idi: int, K_, y_):
        """
        Call to the hook after the meta Cholesky has made a down-date.
        :param idi: current datapoint that the meta Cholesky is processing
        :param K_: pointer to the part of the matrix where the inner Cholesky will be computed
        :param y_: pointer to the part of the solution vector which will be updated next
        :return:
        whether to stop or not
        """
        return False

    def post_chol(self, idi: int, K_, y_):
        """
        Call to the hook after the meta Cholesky has computed the inner Cholesky.
        :param idi: current datapoint that the meta Cholesky is processing
        :param K_: pointer to the part of the matrix where the inner Cholesky has been computed
        :param y_: pointer to the part of the solution vector which has been updated last
        :return:
        whether to stop or not
        """
        return False

    def prepare(self, A, y, block_size: int):
        """
        Gives the hook a chance to initialize before the meta Cholesky starts.
        :param A:
        pointer to the Cholesky buffer
        :param y:
        pointer to the target buffer
        :param block_size:
        the block_size that the Cholesky is going to use
        :return:
        whether to stop or not
        """
        self.A = A
        self.y = y
        self.block_size = block_size
        return False

    def finalize(self):
        self.finished = True
        self._finalize()

    def _finalize(self):
        pass
