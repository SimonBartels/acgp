import scipy.linalg.blas as FB


def fdot(X, Y):
    """
    Dot product X * Y.T making sure that the output is Fortran aligned.
    https://stackoverflow.com/questions/9478791/is-there-an-enhanced-numpy-scipy-dot-method
    :param X: numpy array
    :param Y: numpy array
    :return: X * Y.T
    """
    return FB.dgemm(alpha=1.0, a=X, b=Y, trans_b=True)
