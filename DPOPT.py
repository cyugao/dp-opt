import torch
import numpy as np
import scipy.linalg
from torch.distributions.laplace import Laplace

GRADIENT_STEP = 0
CURVATURE_STEP = 1
FINISHED_STEP = -1


class DPOPT(torch.optim.Optimizer):

    """Differentially Private Optimization Algorithm"""

    def __init__(self, params, opt_params, use_mini_batch=True, line_search=True):

        # defaults = dict(grad_sigma=grad_sigma, hess_sigma=hess_sigma, eps_g=eps_g, eps_H=eps_H, n_dim=n_dim,
        #                 c=c, c1=c1, c2=c2)
        # if nesterov and (momentum <= 0 or dampening != 0):
        #     raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DPOPT, self).__init__(params, opt_params)

        if len(self.param_groups) != 1:
            raise ValueError(
                "DPOPT doesn'i support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.completed = False
        self.rdp_accountant = 0
        self.mini_batch = use_mini_batch
        self.line_search = line_search
        self.eps_g = opt_params["eps_g"]
        self.eps_H = opt_params["eps_H"]
        self.gamma_g_init = opt_params["b_g"] * opt_params["gamma_g_bar"]
        self.qg_scalar = 2 / opt_params["n"] * self.gamma_g_init * opt_params["B_g"]
        self.gamma_H_init_scalar = (
            opt_params["b_H"] * opt_params["t2"] / opt_params["M"]
        )
        self.qH_scalar = 2 / opt_params["n"] * opt_params["G"]

        # keep track of the RDP
        self.rdp_mech = None
        self.grad_evals = 0
        self.hess_evals = 0
        self.grad_evals_total = 0
        self.hess_evals_total = 0

    @torch.no_grad()
    def step(self, closure, hess_closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            hess_closure: A closure that reevaluates the hessian
        """
        loss = None
        # count number of iterations
        # count gradient steps and neg curvature steps
        # mechanism_list = []
        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        n_dim = group["n_dim"]
        c_g = group["c_g"]
        c_H = group["c_H"]
        # c = group["c"]
        # c1 = group["c1"]
        # c2 = group["c2"]
        gamma_g_init = self.gamma_g_init
        gamma_g_bar = group["gamma_g_bar"]
        beta_g = group["beta_g"]
        beta_H = group["beta_H"]
        G = group["G"]
        M = group["M"]
        grad_sensitivity = group["grad_sensitivity"]
        hess_sensitivity = group["hess_sensitivity"]
        b_H = group["b_H"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        # state = self.state[param]
        # state.setdefault('func_evals', 0)
        # state.setdefault('n_iter', 0)

        # evaluate initial f(x)
        param = self._params[0]
        orig_loss = closure()
        loss = float(orig_loss)
        # current_evals = 1
        # state['func_evals'] += 1

        # flat_grad = self._gather_flat_grad()
        self.grad_evals += 1
        # breakpoint()

        # print(grad_sensitivity * self.sigma_g)
        noisy_grad = param.grad + grad_sensitivity * self.sigma_g * torch.randn(n_dim)
        noisy_grad_norm = noisy_grad.norm()
        w_init = param.clone(memory_format=torch.contiguous_format)

        def step_closure(closure, direction, gamma):
            """
            Args:
                closure: A closure that reevaluates the model
                    and returns the loss.
                direction: The direction to take the step
                gamma: The step size
            """
            param.copy_(w_init + gamma * direction)
            return float(closure())

        if noisy_grad_norm > self.eps_g:
            # Gradient step
            if not self.line_search:
                param.add_(noisy_grad, alpha=-1 / G)
                return GRADIENT_STEP, noisy_grad_norm
            # Backtracking line search
            qg_sensitivity = self.qg_scalar * noisy_grad_norm

            closure_qg = (
                lambda gamma: loss
                - step_closure(closure, -noisy_grad, gamma)
                - c_g * gamma * noisy_grad_norm * noisy_grad_norm
            )

            gamma = self.svt_line_search(
                closure_qg,
                qg_sensitivity,
                beta_g,
                gamma_g_init,
                gamma_g_bar,
                self.lambda_svt,
            )
            # ic(gamma)
            param.copy_(w_init - gamma * noisy_grad)
            return GRADIENT_STEP, noisy_grad_norm
        else:
            print("curvature step")
            noisy_hess = (
                hess_closure()
                + hess_sensitivity * self.sigma_H * self.random_symmetric_matrix(n_dim)
            )
            lam, v = self.smallest_eig(noisy_hess)
            lam_abs = abs(lam)
            if torch.dot(v, noisy_grad) > 0:
                v = -v
            self.hess_evals += 1

            if lam > -self.eps_H:
                self.completed = True
                return FINISHED_STEP, lam

            # Negative curvature step
            if not self.line_search:
                param.add_(v, alpha=2 * lam_abs / M)
                return CURVATURE_STEP, lam

            # Backtracking line search
            gamma_H_init = self.gamma_H_init_scalar * lam_abs
            gamma_H_bar = gamma_H_init / b_H

            closure_qH = (
                lambda gamma: loss
                - step_closure(closure, v, gamma)
                - 0.5 * c_H * gamma * gamma * lam_abs
            )
            qH_sensitivity = self.qH_scalar * gamma_H_init
            # ic(gamma_H_bar, gamma_H_init, qH_sensitivity)
            gamma = self.svt_line_search(
                closure_qH,
                qH_sensitivity,
                beta_H,
                gamma_H_init,
                gamma_H_bar,
                self.lambda_svt,
            )
            param.copy_(w_init + gamma * v)
            return CURVATURE_STEP, lam
        # accumulate RDP
        # self.mech = compose(mechanism_list, coeff_list, RDP_compose_only=True)

    def svt_line_search(
        self, closure_q, q_sensitivity, beta, gamma_init, gamma_bar, lambda_svt
    ):
        """
        Performs a line search on the given closure
        """
        gamma = gamma_init
        xi = Laplace(0, 2 * lambda_svt * q_sensitivity).sample()
        i_max = int(np.ceil(np.log(gamma_bar / gamma_init) / np.log(beta)))

        lap = Laplace(0, 4 * lambda_svt * q_sensitivity)
        for _ in range(i_max):
            q = closure_q(gamma) + lap.sample()
            if q > xi:
                return gamma
            gamma = gamma * beta

        return gamma_bar

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

    # def compose_rdp(self, frac=0.5):
    #     gm_grad = ExactGaussianMechanism(
    #         self.sigma_g / self.grad_sensitivity, name="GM_grad"
    #     )
    #     gm_hess = ExactGaussianMechanism(
    #         self.sigma_H / self.hess_sensitivity, name="GM_hess"
    #     )
    #     SVT = PureDP_Mechanism(eps=0.1, name="SVT_line_search")  # TODO fix this
    #     gm_grad.replace_one = True
    #     gm_hess.replace_one = True

    #     if self.mini_batch:
    #         gm_grad = subsample(gm_grad, frac, improved_bound_flag=True)
    #         gm_hess = subsample(gm_hess, frac, improved_bound_flag=True)

    #     return compose([gm_grad, gm_hess], [self.grad_evals, self.hess_evals])

    def new_epoch_rho(self, rho_0):
        if self.line_search:
            self.sigma_g = self.sigma_H = self.lambda_svt = (3 / (2 * rho_0)) ** 0.5
        else:
            self.sigma_g = self.sigma_H = (1 / rho_0) ** 0.5
            self.lambda_svt = None
        return self.sigma_g
            

    def new_epoch(self, sigma_g, sigma_H, lambda_svt):
        self.sigma_g = sigma_g
        self.sigma_H = sigma_H
        self.lambda_svt = lambda_svt

    def set_sosp_params(self, eps_g, eps_H):
        self.eps_g = eps_g
        self.eps_H = eps_H
