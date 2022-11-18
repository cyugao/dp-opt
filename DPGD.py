import scipy

import torch
import numpy as np


class DPGD(torch.optim.Optimizer):

    """Differentially Private Optimization Algorithm"""

    def __init__(
        self, params, opt_params, eps, delta, initial_gap=None, use_mini_batch=True
    ):

        # defaults = dict(grad_sigma=grad_sigma, hess_sigma=hess_sigma, eps_g=eps_g, eps_H=eps_H, n_dim=n_dim,
        #                 c=c, c1=c1, c2=c2)
        # if nesterov and (momentum <= 0 or dampening != 0):
        #     raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DPGD, self).__init__(params, opt_params)

        if len(self.param_groups) != 1:
            raise ValueError(
                "DPOPT doesn'i support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.completed = False
        self.rdp_accountant = 0
        self.mini_batch = use_mini_batch
        self.eps = eps
        self.delta = delta
        # sigma_g = 2G / n * sqrt(T/phi)
        self.alpha = opt_params["alpha"]
        self.eps_H = opt_params["eps_H"]
        self.B = opt_params["B_f"]
        self.B_g = opt_params["B_g"]
        self.G = opt_params["G"]
        self.M = opt_params["M"]
        self.n_dim = opt_params["n_dim"]
        self.I = np.eye(self.n_dim)
        self.i = 0
        self.C1 = 0.1
        # C1 and large enough constant c? c1, c2? xi?
        self.c = 0.7
        self.c1 = 1
        self.c2 = 1
        self.c3 = 1
        self.xi = 1
        # chi = max(1, C1 * log (n_dim G B / (M alpha xi)))
        # phi = sqrt(alpha^3 / M) * chi^(-3) * c^(-5)
        # r = alpha * chi^(-3) * c^(-6)
        # gamma = G * chi * c / sqrt(M alpha)
        self.chi = max(
            1,
            self.C1
            * np.log(self.n_dim * self.G * self.B / (self.M * self.alpha * self.xi)),
        )
        self.phi = np.sqrt(self.alpha**3 / self.M) * self.chi ** (-3) * self.c ** (-5)
        self.r = self.alpha * self.chi ** (-3) * self.c ** (-6)
        self.gamma = self.G * self.chi * self.c / np.sqrt(self.M * self.alpha)
        # T = gap / min(alpha^2/(4M), phi / gamma)
        self.T = int(
            np.ceil(
                initial_gap / min(self.alpha**2 / (4 * self.M), self.phi / self.gamma)
            )
        )
        # constant sigma_0^2 = log(1/delta) * T * / (n^2 eps^2)
        # self.sigma_0 = np.log(1 / self.delta) * self.T / (self.n_dim * self.eps)
        self.sigma0 = np.sqrt(np.log(1 / self.delta) * self.T) / self.eps
        self.sigma = self.c2**0.5 * self.sigma0
        self.sigma1 = self.c2**0.5 * self.sigma0
        self.sigma2 = self.c3**0.5 * self.sigma0

        self.grad_evals = 0
        self.hess_evals = 0

    @torch.no_grad()
    def step(self, closure, hess_closure):
        self.i += 1
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        hess_sensitivity = group["hess_sensitivity"]
        grad_sensitivity = group["grad_sensitivity"]
        n_dim = group["n_dim"]

        param = self._params[0]
        self.grad_evals += 1
        noisy_grad = param.grad + grad_sensitivity * self.sigma1 * torch.randn(n_dim)
        if noisy_grad.norm() < self.alpha:
            noisy_hess = (
                hess_closure()
                + hess_sensitivity * self.sigma2 * self.random_symmetric_matrix(n_dim)
            )
            self.hess_evals += 1
            lam, v = self.smallest_eig(noisy_hess)
            if lam > -self.eps_H:
                self.completed = True
                return
        noisy_grad = param.grad + grad_sensitivity * self.sigma * torch.randn(n_dim)
        param.add_(-noisy_grad / self.G)

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
