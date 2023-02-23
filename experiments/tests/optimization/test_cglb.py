import torch

import unittest
from utils.data.load_dataset import get_train_test_dataset
from gpytorch.kernels import MaternKernel
from hyperparameter_tuning.utils.gpytorch.models.cglb import CGLB
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import OPTIMIZE_INDUCING_INPUTS, NUM_INDUCING_INPUTS, SELECTION_SCHEME, \
    CONDITIONAL_VARIANCE, MAX_NUM_CG_STEPS
from hyperparameter_tuning.utils.gpytorch.scipy import Scipy


class CGLBTestCase(unittest.TestCase):
    def test_gradient(self):
        device = "cpu"
        X, y, _, _ = get_train_test_dataset("wilson_pumadyn32nm")
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)
        X = X.to(device)
        y = y.to(device)
        k = MaternKernel()
        k = k.to(device)
        k.train()
        sn2 = lambda: torch.tensor(1, dtype=torch.float64, device=device)
        mu = lambda X: torch.zeros(X.shape[0], dtype=torch.float64, device=device)
        cglb = CGLB(X, y, k, sn2, mu, args={OPTIMIZE_INDUCING_INPUTS: True, NUM_INDUCING_INPUTS: 2,
                                            SELECTION_SCHEME: CONDITIONAL_VARIANCE, MAX_NUM_CG_STEPS: 100},
                    device=device)

        loss = cglb.create_loss_closure()

        if False:
            t = time()
            tt = thread_time()
            l = loss()
            print(f"time: {time() - t}")
            print(f"thread_time: {thread_time() - tt}")

            t = time()
            tt = thread_time()
            l.backward()
            print(f"time: {time() - t}")
            print(f"thread_time: {thread_time() - tt}")

        variables = tuple([v for _, v in cglb.get_named_tunable_parameters()])

        if False:
            t = time()
            tt = thread_time()
            l = loss()
            print(f"time: {time() - t}")
            print(f"thread_time: {thread_time() - tt}")

            t = time()
            tt = thread_time()
            grads = torch.autograd.grad(l, variables)
            print(f"time: {time() - t}")
            print(f"thread_time: {thread_time() - tt}")

        # exit()

        x0 = Scipy.pack(variables)

        def _torch_eval(x):
            values = Scipy.unpack(variables, x)
            Scipy.assign(variables, values)
            return loss()

        torch.autograd.gradcheck(_torch_eval, x0)


if __name__ == '__main__':
    unittest.main()
