import inspect
import numpy as np

from utils.mem_efficient_kernels.fdot import fdot


class IsotropicKernel:
    def __init__(self):
        self.name = type(self).__name__

    def initialize(self, log_var=0., log_ls2=0.):
        self.var = np.exp(log_var)  # amplitude
        self.ls = np.exp(2 * log_ls2)  # we store the squared length-scale
        return self

    def initialize_from_parser(self, parsed_arguments):
        d = {}
        args = inspect.signature(self.initialize).parameters.keys()
        for arg in args:
            # a = arg.replace('_', '-')
            d[arg] = parsed_arguments[self.name + "_" + arg]
        return self.initialize(**d)

    def get_parameter_dictionary(self):
        d = {"log_var": np.log(self.var), "log_ls2": np.log(self.ls) / 2}
        assert(set(d.keys()) == set(self.get_default_parameter_dictionary().keys()))
        return d

    def get_default_parameter_dictionary(self):
        d = {}
        parameter_dict = inspect.signature(self.initialize).parameters
        args = parameter_dict.keys()
        for arg in args:
            #a = arg.replace('_', '-')
            d[arg] = parameter_dict[arg].default
        return d

    def add_parameters_to_parser(self, parser):
        parameter_dict = inspect.signature(self.initialize).parameters
        args = parameter_dict.keys()
        for arg in args:
            short = ""
            for l in arg.split('_'):
                short += l[0]
            parser.add_argument("-%s-%s" % (self.name, short),
                                "--%s-%s" % (self.name, arg.replace('_', '-')),
                                type=type(parameter_dict[arg].default), default=parameter_dict[arg].default)

    def generate_command_string(self):
        d = self.get_parameter_dictionary()
        return ''.join([' --' + self.name + '-' + k.replace('_', '-') + ' ' + str(d[k]) for k in d.keys()])

    def K(self, X, X2=None):
        K = self._K(self._get_neg_dist(X, X2))
        K *= self.var
        return K

    def Kdiag(self, X):
        return self.var * np.ones(X.shape[0])

    def _K(self, dist):
        raise NotImplementedError("abstract method")

    def _get_neg_dist(self, X, X2=None):
        Xsq = np.sum(np.square(X), 1)
        if X2 is None:
            K = 2 * fdot(X, X)  # X * X.T
            K -= Xsq[:, None]
            K -= Xsq[None, :]
            np.fill_diagonal(K, 0.)
        else:
            X2sq = np.sum(np.square(X2), 1)
            K = 2 * fdot(X, X2) - (Xsq[:, None] + X2sq[None, :])
        K /= self.ls
        return K


class RBF(IsotropicKernel):
    def _K(self, K):
        K /= 2.
        np.exp(K, out=K)
        return K


class OU(IsotropicKernel):
    def _K(self, K):
        np.negative(K, out=K)  # make distances positive
        K[K < 0] = 0
        np.sqrt(K, out=K)  # this also takes the root of the length scale
        np.negative(K, out=K)
        np.exp(K, out=K)
        return K


class Delta(IsotropicKernel):
    def K(self, X, X2=None):
        if X2 is None:
            return self.var * np.eye(X.shape[0], order='F')
        else:
            return np.zeros((X.shape[0], X2.shape[0]), order='F')


class Zero(IsotropicKernel):
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0]), order='F')

    def Kdiag(self, X):
        return np.zeros(X.shape[0])
