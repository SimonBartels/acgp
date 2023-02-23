import numpy as np


def get_log_relative_errors_save(approximate_array, exact_value):
    """
    Returns the (signed) relative error of the approximate values such that a relative error of 0 remains 0.
    :param approximate_array:
    :param exact_value:
    :return:
    """
    a = approximate_array - exact_value
    #a /= exact_value
    b = a[np.abs(a)!=0.]
    a[np.abs(a)!=0.] = np.sign(b) * np.log(np.abs(b))
    return a
