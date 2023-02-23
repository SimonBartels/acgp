from time import time, thread_time
import torch

ls = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
amp = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

N = 3000
D = 10
X = torch.randn([N, D])
y = torch.randn([N, 1])


def _get_neg_dist(X: torch.Tensor, X2: torch.Tensor = None):
    Xsq = torch.sum(torch.square(X), dim=1)
    if X2 is None:
        K = 2. * X @ X.T
        K = K - Xsq[:, None]
        K = K - Xsq[None, :]
        #np.fill_diagonal(K, 0.)
    else:
        raise NotImplementedError()
        X2sq = np.sum(np.square(X2), 1)
        K = 2 * fdot(X, X2) - (Xsq[:, None] + X2sq[None, :])
    K = K / ls
    return K


def k(X, X2=None):
    return torch.exp(_get_neg_dist(X, X2) + amp)


sn2 = 1e-2
K = k(X) + sn2 * torch.eye(N)
A = torch.linalg.cholesky(K)

if False:
    class MyDiag(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B):
            return torch.diag(K)

        @staticmethod
        def backward(ctx, grad_outputs):
            print(grad_outputs.shape)
            with torch.no_grad():
                return torch.diag(grad_outputs)

    fx = torch.sum(MyDiag.apply(K))
    fx.backward()
    print(ls.grad)
    print(amp.grad)
    ls.grad = torch.zeros_like(ls.grad)
    amp.grad = torch.zeros_like(amp.grad)

    fx = torch.sum(torch.diag(k(X) + sn2 * torch.eye(N)))
    fx.backward()
    print(ls.grad)
    print(amp.grad)
    #exit()


@torch.no_grad()
def Phi(A):
    # TODO: Use in-place operation?
    return torch.tril(A - torch.diag(torch.diag(A)) / 2)


class DiagChol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, K):
        return torch.diag(A)

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.no_grad():
            print(grad_outputs.shape)
            #S = Phi(A.T @ torch.diag(grad_outputs))
            s = torch.mul(torch.diag(A), grad_outputs).reshape(-1, 1)  # O(N)
            diagMul = torch.ones([A.shape[0], 1], dtype=torch.float64) @ s.T  # O(N^2)
            Kinv = torch.cholesky_inverse(torch.mul(diagMul, A))
            return Phi(Kinv)

            S = torch.diag(torch.mul(torch.diag(A), grad_outputs)) / 2
            # TODO: make below operation in-place
            # WHAT THE HELL?! WHY ON EARTH IS TORCH CLONING THE COEFFICIENT MATRIX!!!
            S = torch.triangular_solve(S, A, upper=False, transpose=True).solution.T
            # TODO: need to find a generic way to make this a numpy array if necessary
            # self.blaswrapper.solve_triangular_inplace(L, S, transpose_b=True, lower=True)
            torch.triangular_solve(S, A, upper=False, transpose=True, out=(S, A))
            # self.blaswrapper.solve_triangular_inplace(L, S, transpose_b=False, lower=True)
            grad = Phi(S + S.T)
            return grad  # , None


logdet = torch.sum(torch.log(DiagChol.apply(K)))
t = time()
tt = thread_time()
logdet.backward()
print(f"time: {time() - t}")
print(f"thread_time: {thread_time() - tt}")
print(ls.grad)
print(amp.grad)
ls.grad = torch.zeros_like(ls.grad)
amp.grad = torch.zeros_like(amp.grad)

fx = torch.sum(torch.log(torch.diag(torch.linalg.cholesky(k(X) + sn2 * torch.eye(N)))))
t = time()
tt = thread_time()
fx.backward()
print(f"time: {time() - t}")
print(f"thread_time: {thread_time() - tt}")
print(ls.grad)
print(amp.grad)
