import warnings
import scipy

import torch
import numpy as np


class DPTR(torch.optim.Optimizer):

    """Differentially Private Optimization Algorithm"""

    def __init__(self, params, opt_params, rho=1, initial_gap=0.5):

        # defaults = dict(grad_sigma=grad_sigma, hess_sigma=hess_sigma, eps_g=eps_g, eps_H=eps_H, n_dim=n_dim,
        #                 c=c, c1=c1, c2=c2)
        # if nesterov and (momentum <= 0 or dampening != 0):
        #     raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DPTR, self).__init__(params, opt_params)

        if len(self.param_groups) != 1:
            raise ValueError(
                "DPOPT doesn'i support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.completed = False
        self.rdp_accountant = 0
        # self.mini_batch = use_mini_batch
        # sigma_g = 2G / n * sqrt(T/phi)
        self.alpha = opt_params["alpha"]
        self.M = opt_params["M"]
        self.n_dim = opt_params["n_dim"]
        self.I = np.eye(self.n_dim)
        self.tr_radius = opt_params["alpha"] / opt_params["M"]
        self.T = int(np.ceil(6 * self.M**0.5 * initial_gap / self.alpha**1.5))
        self.sigma_g = (self.T / rho) ** 0.5
        self.sigma_H = (self.T * self.n_dim / rho) ** 0.5
        self.i = 0

    @torch.no_grad()
    def step(self, closure, hess_closure, test_sosp=True):
        self.i += 1
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        hess_sensitivity = group["hess_sensitivity"]
        grad_sensitivity = group["grad_sensitivity"]
        n_dim = group["n_dim"]

        param = self._params[0]
        noisy_grad = param.grad + grad_sensitivity * self.sigma_g * torch.randn(n_dim)
        noisy_hess = (
            hess_closure()
            + hess_sensitivity * self.sigma_H * self.random_symmetric_matrix(n_dim)
        )
        if test_sosp and noisy_grad.norm() < self.alpha:
            lam, v = self.smallest_eig(noisy_hess)
            if lam > -((self.alpha * self.M) ** 0.5):
                self.completed = True
                return
            
        # update, dual = self.solve_qcqp(noisy_hess, noisy_grad, noisy_grad.norm())
        update, dual = self.solve_qcqp(noisy_hess, noisy_grad, self.tr_radius)
        # print(f"dual: {dual:.5f}")
        param.add_(torch.from_numpy(update))
        if dual < (self.alpha * self.M) ** 0.5:
            self.completed = True

    def solve_qcqp(self, H, g, r):
        # if positive definite, check the easy case
        lam1, _ = self.smallest_eig(H)
        if lam1 > 0:
            x = np.linalg.solve(H, -g)
            if np.linalg.norm(x) <= r:
                return x, 0

        # First solve the following Nonsymmetric Eigenvalue Problem
        # form A = [H, -I; -gg'/r^2, H]
        n_dim = H.shape[0]
        A = np.concatenate(
            (
                np.concatenate((H, -np.identity(n_dim)), axis=1),
                np.concatenate((-np.outer(g, g) / r**2, H), axis=1),
            ),
            axis=0,
        )
        w, v = np.linalg.eig(A)
        # get the minimum real eigenvalue in w
        lam = w[np.isreal(w)].real.min()
        lam = -lam
        # solve (H+lam*I)x = -g
        if lam1 > lam:
            # hard case
            warnings.warn(f"lam1={lam1:1.2f} > lam={lam:.4f}, hard case!")
        x = np.linalg.solve(H + lam * np.identity(n_dim), -g)
        return x, lam

    def smallest_eig(self, H):
        """Compute the smallest (eigenvalue, eigenvector) pair of a complex Hermitian or real symmetric matrix.

        Note:
            For real symmetric matrices, use `scipy.linalg.eigh` instead of `scipy.linalg.eig` for better performance.

        Returns:
            i: the smallest eigenvalue
            v: corresponding eigenvector of shape (dim, 1)

        """
        i, v = scipy.linalg.eigh(H, subset_by_index=(0, 0))
        return i[0], torch.from_numpy(v.ravel())

    def random_symmetric_matrix(self, N):
        """Generate a random symmetric metrix, of which each entry is unit Gaussian noise."""
        a = torch.randn(N, N)
        return torch.tril(a) + torch.tril(a, -1).T
