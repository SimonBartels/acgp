import torch
import weakref


def get_custom_log_det_plus_quad(L, alpha, result, subset_size):
    # L_ = weakref.ref(L)
    # alpha_ = weakref.ref(alpha)
    # result_ = weakref.ref(result)
    class Custom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, K, y):
            return result

        @staticmethod
        def backward(ctx, grad_outputs):
            with torch.no_grad():
                subset_size_ = subset_size[0]
                L_ = L[:subset_size_, :subset_size_]
                alpha_ = alpha[:subset_size_, :]
                # Kinv = torch.cholesky_inverse(L) * grad_outputs
                # matrix_grad = Kinv - alpha * grad_outputs @ alpha.T
                # vec_grad = alpha * (2 * grad_outputs)
                # return matrix_grad, vec_grad
                Kinv = torch.cholesky_inverse(L_)
                Kinv -= alpha_ @ alpha_.T
                Kinv *= grad_outputs
                return Kinv, alpha_ * (2 * grad_outputs)

    return Custom


def _Phi(X):
    return torch.tril(X) - torch.diag(torch.diag(X)) / 2
