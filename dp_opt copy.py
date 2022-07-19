from datetime import datetime
import os
import time

import warnings
import scipy.linalg
import numpy as np
import pandas as pd
from torch.distributions.laplace import Laplace
import torch
from numpy.random import default_rng
from icecream import ic

import wandb

WANDB_PROJECT = "dp-opt-exp-new"

# In[15]:

df_covtype = pd.read_csv("data/covtype.csv")
df_covtype.head()
normalize_df = lambda df: (df - df.mean()) / df.std()
# Standardize numerical columns
df_covtype.iloc[:, :10] = normalize_df(df_covtype.iloc[:, :10])


# In[16]:

# only keep cover type 1 and 2 for binary classification
df_covtype = df_covtype[df_covtype["Cover_Type"] <= 2]

X, y = df_covtype.iloc[:, :-1].to_numpy(), df_covtype.iloc[:, -1].to_numpy()
# Change labels to +1 and -1
y[y == 2] = -1
X = torch.Tensor(X)
y = torch.Tensor(y)

n, n_dim = X.shape
X.shape, y.shape


class ERM(torch.nn.Module):
    def __init__(self, input_dim, regularizer, reg_coef=1e-3):
        super(ERM, self).__init__()
        self.input_dim = input_dim
        self.regularizer = regularizer
        self.reg_coef = reg_coef

        self.w = torch.nn.Parameter(torch.randn(self.input_dim))

    def forward(self, X, y, w=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if w is None:
            w = self.w
        return torch.log1p(
            torch.exp(-y * (X @ w))
        ).mean() + self.reg_coef * self.regularizer(w)

    # def string(self):
    #     """
    #     Just like any class in Python, you can also define custom method on PyTorch modules
    #     """
    #     return f'loss = {self.w.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


# In[20]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def regularizer(w):
    w2 = w * w
    return (w2 / (1 + w2)).sum()


def erm_training_loop(
    dataloader, model, optimizer, scheduler=None, epochs=3, eval_on_train=True
):
    size = len(dataloader.dataset)
    for i in range(epochs):
        print(f"\nEpoch {i+1}\n-------------------------------")
        for batch, (XX, yy) in enumerate(dataloader):
            XX, yy = XX.to(device), yy.to(device)

            # Compute prediction error
            loss = model(XX, yy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 3 == 0:
                loss, current = loss.item(), batch * len(XX)
                if eval_on_train:
                    train_loss = model(X, y).item()
                    print(
                        f"loss: {loss:>5f}, loss on the whole dataset: {train_loss:>5f}  [{current:>5d}/{size:>5d}]"
                    )
                else:
                    print(f"loss: {loss:>5f} [{current:>5d}/{size:>5d}]")
        if scheduler:
            scheduler.step


# ### Helper Functions

# ## DPOPT

# For RDP, eps is a function of gamma $$\varepsilon=\varepsilon(\gamma)$$

# ## Quick example of autodp


from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism
from autodp.transformer_zoo import Composition, ComposeGaussian, AmplificationBySampling


# In[164]:

# gm_grad = ExactGaussianMechanism(0.1, name="GM_grad")
# gm_hess = ExactGaussianMechanism(0.2, name="GM_hess")
# subsample = AmplificationBySampling(PoissonSampling=False)
# SVT = PureDP_Mechanism(eps=0.1, name="SVT_line_search")
# compose = Composition()


# In[221]:
class DPTR(torch.optim.Optimizer):

    """Differentially Private Optimization Algorithm"""

    def __init__(self, params, opt_params, rho=1, use_mini_batch=True):

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
        self.complete = False
        self.rdp_accountant = 0
        self.mini_batch = use_mini_batch
        self.T = opt_params["T"]
        # sigma_g = 2G / n * sqrt(T/phi)
        self.alpha = opt_params["alpha"]
        self.M = opt_params["M"]
        self.n_dim = opt_params["n_dim"]
        self.I = np.eye(self.n_dim)
        self.tr_radius = opt_params["alpha"] / opt_params["M"]
        self.sigma_g = 2 * opt_params["G"] / opt_params["n"] * (self.T / rho) ** 0.5
        self.sigma_H = (
            2
            * opt_params["M"]
            / opt_params["n"]
            * (self.T * opt_params["n_dim"] / rho) ** 0.5
        )
        self.i = 0

    @torch.no_grad()
    def step(self, closure, hess_closure):
        self.i += 1
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        hess_sensitivity = group["hess_sensitivity"]
        grad_sensitivity = group["grad_sensitivity"]

        param = self._params[0]
        orig_loss = closure()
        noisy_grad = param.grad + grad_sensitivity * self.sigma_g * torch.randn(n_dim)
        noisy_hess = (
            hess_closure()
            + hess_sensitivity * self.sigma_H * self.random_symmetric_matrix(n_dim)
        )

        # update, dual = self.solve_qcqp(noisy_hess, noisy_grad, noisy_grad.norm())
        update, dual = self.solve_qcqp(noisy_hess, noisy_grad, self.tr_radius)
        # print(f"dual: {dual:.5f}")
        param.add_(torch.from_numpy(update))
        if dual < (self.alpha * self.M) ** 0.5:
            self.complete = True

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

    # def solve_qcqp(self, H, g, r):
    #     x = cp.Variable(self.n_dim)
    #     r2 = r * r
    #     constraint = 0.5 * cp.quad_form(x, self.I) <= 0.5 * r2
    #     p = cp.Problem(
    #         cp.Minimize(0.5 * cp.quad_form(x, H) + g.T @ x),
    #         [constraint],
    #     )
    #     # [cp.atoms.pnorm(vec(x), 2) <= r])
    #     primal_result = p.solve()

    #     if p.status is not cp.OPTIMAL:
    #         raise ValueError("Failed to solve the subproblem")

    #     # return solution and dual
    #     return x.value, constraint.dual_value

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
        self.complete = False
        self.rdp_accountant = 0
        self.mini_batch = use_mini_batch
        self.line_search = line_search
        self.gamma_g_init = opt_params["b_g"] * opt_params["gamma_g_bar"]
        self.qg_scalar = 2 / opt_params["n"] * self.gamma_g_init * opt_params["G"]
        self.gamma_H_init_scalar = (
            opt_params["b_H"] * opt_params["t2"] / opt_params["M"]
        )
        self.qH_scalar = (
            opt_params["b_H"]
            * 2
            / opt_params["n"]
            * self.gamma_g_init
            * opt_params["G"]
        )

        # keep track of the RDP
        self.rdp_mech = None
        self.subsample = AmplificationBySampling(PoissonSampling=False)
        self.grad_evals = 0
        self.hess_evals = 0

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
        eps_g = group["eps_g"]
        eps_H = group["eps_H"]
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

        if noisy_grad_norm > eps_g:
            # Gradient step
            if not self.line_search:
                param.add_(noisy_grad, alpha=-1 / G)
                return
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
        else:
            print("curvature step")
            noisy_hess = (
                hess_closure()
                + hess_sensitivity * self.sigma_H * self.random_symmetric_matrix(n_dim)
            )
            i, v = self.smallest_eig(noisy_hess)
            self.hess_evals += 1

            if i > -eps_H:
                self.complete = True
                return

            # Negative curvature step
            if not self.line_search:
                param.add_(v, alpha=2 * -i / M)
                return

            # Backtracking line search
            gamma_H_init = self.gamma_H_init_scalar * -i

            closure_qH = (
                lambda gamma: loss
                - step_closure(closure, v, gamma)
                - 0.5 * c_H * gamma * gamma * -i
            )
            qH_sensitivity = self.qH_scalar * -i

            gamma = self.svt_line_search(
                closure_qH,
                qH_sensitivity,
                beta_H,
                gamma_H_init,
                gamma_g_bar,
                self.lambda_svt,
            )
            param.copy_(w_init + gamma * v)

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

    def new_epoch(self, sigma_g, sigma_H, lambda_svt):
        # self.grad_evals = 0
        # self.hess_evals = 0
        self.sigma_g = sigma_g
        self.sigma_H = sigma_H
        self.lambda_svt = lambda_svt


#     def accumulate_gaussian_rdp(self, sigma, sensitivity):
#         self.rdp_accountant += 0.5 * (sensitivity / sigma) ** 2

#     def get_rdp_accountant(self):
#         return self.rdp_accountant


def check_and_compute_params(opt_params):
    if opt_params["checked"]:
        return
    n = opt_params["n"]
    B_g = opt_params["B_g"]
    B_H = opt_params["B_H"]
    eps_g = opt_params["eps_g"]
    eps_H = opt_params["eps_H"]
    n_dim = opt_params["n_dim"]
    G = opt_params["G"]
    M = opt_params["M"]
    c = opt_params["c"]
    c1 = opt_params["c1"]
    c2 = opt_params["c2"]
    beta_g = opt_params["beta_g"]
    c_g = opt_params["c_g"]
    b_g = opt_params["b_g"]
    beta_H = opt_params["beta_H"]
    c_H = opt_params["c_H"]
    b_H = opt_params["b_H"]

    opt_params["f_sensitivity"] = opt_params["loss_sensitivity"] / n
    opt_params["grad_sensitivity"] = 2 * B_g / n
    opt_params["hess_sensitivity"] = 2 * B_H * n_dim**0.5 / n

    assert c1 + c_g < 1
    opt_params["gamma_g_bar"] = 2 * (1 - c1 - c_g) / G

    delta = 0.5 * (1 - c - c_H) ** 2 - 2 / 3 * c2
    assert delta > 0
    opt_params["t2"] = 1.5 * (1 - c - c_H) + 3 * delta**0.5

    gamma_g_bar = opt_params["gamma_g_bar"]
    t2 = opt_params["t2"]
    term1_ls = beta_g * c_g * gamma_g_bar * eps_g * eps_g
    term2_ls = 0.5 * c_H * beta_H * beta_H * t2 * t2 * eps_H / (M * M)
    opt_params["min_decrease_ls"] = min(term1_ls, term2_ls)

    assert c1 < 1 / 2
    assert c2 + c < 1 / 3
    term1 = (1 - 2 * c1) / (2 * G) * (eps_g * eps_g)
    term2 = 2 * (1 / 3 - c2 - c) * eps_H**3 / (M * M)
    opt_params["min_decrease"] = min(term1, term2)

    opt_params["checked"] = True


def dp_estimate_T(
    full_loss_closure, opt_params, sigma_f, rng=2022, gap=None, line_search=True
):
    assert opt_params["checked"]
    if not gap:
        gap = full_loss_closure() - opt_params["lower_bound"]
        gap += sigma_f * opt_params["f_sensitivity"] * abs(rng.normal(1))
    if line_search:
        decrease = opt_params["min_decrease_ls"]
    else:
        decrease = opt_params["min_decrease"]

    return int(np.ceil(gap / decrease))


## In[171]:

# track order α = {1.25, 1.5, 1.75, . . . , 9.75, 10, 16, 32, 64}

alphas = torch.cat((torch.arange(1.25, 10.1, 0.25), torch.Tensor([16, 32, 64])))


# def get_rdp(mech, alphas=alphas):
#     return [mech.get_RDP(gamma).item() for gamma in alphas]


def dp_erm_training_loop(
    X,
    y,
    model,
    optimizer,
    opt_params,
    alphas=alphas,
    line_search=True,
    epochs=3,
    eval_on_train=True,
):
    check_and_compute_params(opt_params)
    rng = default_rng(22)
    full_loss_closure = lambda: model(X, y).item()
    n = len(y)
    sigma_f = 0.1  # TODO determine this

    gm_f = ExactGaussianMechanism(sigma_f / opt_params["loss_sensitivity"], name="GM_f")
    gm_f.replace_one = True
    mechanism_lst = []

    batch_frac = 0.5
    mech = None

    for i in range(epochs):

        est_T = dp_estimate_T(full_loss_closure, opt_params, sigma_f)
        # mech_full = gm_f if i == 0 else compose([gm_f, mech], [i + 1, 1])
        # eps_alphas = get_rdp(mech_full, alphas)
        # print(f"eps: {eps_alphas[:5]} for gamma={alphas[:5]}")

        # determine sigmas and batch_size
        sigma_g, sigma_H, lambda_svt = 0.1, 0.1, 0.1
        batch_size = 5000
        batch_frac = batch_size / n

        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        # T = est_T // 4
        T = 10
        print(f"Epoch {i+1}: {T} iterations \n-------------------------------")
        for i in range(T):
            indices = rng.choice(n, size=batch_size)
            XX, yy = X[indices], y[indices]
            XX, yy = XX.to(device), yy.to(device)

            # Compute prediction error
            loss = model(XX, yy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            hess_closure = lambda: torch.autograd.functional.hessian(
                lambda w: model(XX, yy, w), model.w
            )
            optimizer.step(lambda: model(XX, yy), hess_closure)
            if optimizer.complete:
                print("Optimization complete!")
                break
            if i % 5 == 0:
                print(f"Iteration {i}: {loss.item()}")
        # print(batch_frac)
        # mech_t = optimizer.compose_rdp(batch_frac)
        # mechanism_lst.append(mech_t)
        # mech = mech_t if i == 0 else compose([mech, mech_t], [1, 1])

        # loss, loss.item()
        if eval_on_train:
            train_loss = full_loss_closure()
            print(
                f"batch size = {batch_size}, loss: {loss:>5f}, loss on the whole dataset: {train_loss:>5f}"
            )
        else:
            print(f"batch size = {batch_size}, loss: {loss:>5f}")
    # mech_full = compose([gm_f, mech], [i + 1, 1])
    # return mechanism_lst, mech_full


# mechanism_lst, mech = dp_erm_training_loop(
#     X, y, model, optimizer, opt_params, alphas=alphas, epochs=30, eval_on_train=True
# )
# dp_erm_training_loop(
#     X, y, model, optimizer, opt_params, alphas=alphas, epochs=3, eval_on_train=True
# )#%%# %%
# torch.manual_seed(22)
# model = ERM(n_dim, regularizer)
# optimizer = DPOPT(model.parameters(), opt_params, line_search=True)
# eval_on_train = True

# hess = torch.autograd.functional.hessian(lambda w: model(X, y, w), model.w)
# print(scipy.linalg.norm(hess, 2))  # 0.3


def dpopt_exp(
    opt_params,
    initial_gap=None,
    rho=1,
    max_iter=1000,
    init_T=500,
    line_search=True,
    print_every=50,
    seed=22,
    wandb_on=True,
):
    n = len(y)
    wandb.init(
        project=WANDB_PROJECT,
        group=f"eps_g={eps_g_target},seed={seed}",
        config={
            "method": "dpopt",
            "seed": seed,
            "rho": rho,
            "eps": rdp2dp(rho, 1 / n),
            "max_iter": max_iter,
            "line_search": line_search,
            "eps_g": opt_params["eps_g"],
            "eps_H": opt_params["eps_H"],
            "params": opt_params,
        },
        mode="online" if wandb_on else "disabled",
    )
    torch.manual_seed(seed)
    model = ERM(opt_params["n_dim"], regularizer)
    optimizer = DPOPT(model.parameters(), opt_params, line_search=line_search)
    rng = default_rng(seed)
    full_loss_closure = lambda: model(X, y).item()

    # gm_f = ExactGaussianMechanism(sigma_f / opt_params["loss_sensitivity"], name="GM_f")
    # gm_f.replace_one = True
    # mechanism_lst = []

    # batch_frac = 0.5
    # mech = None
    rho_f = rho / max_iter
    sigma_f = (1 / (2 * rho_f)) ** 0.5
    sigma_f_unscaled = sigma_f * opt_params["f_sensitivity"]
    print(f"sigma_f: {sigma_f}")
    est_T = dp_estimate_T(full_loss_closure, opt_params, sigma_f, rng, gap=initial_gap)
    rho -= rho_f
    print(f"Estimated T={est_T}")
    # exit()
    T = est_T

    # mech_full = gm_f if i == 0 else compose([gm_f, mech], [i + 1, 1])
    # eps_alphas = get_rdp(mech_full, alphas)
    # print(f"eps: {eps_alphas[:5]} for gamma={alphas[:5]}")

    # determine sigmas and batch_size
    # sigma_g, sigma_H, lambda_svt = 0.1, 0.1, 0.1
    # batch_size = 5000
    # batch_frac = batch_size / n

    # T = est_T // 4
    # i = 0
    # print(f"Epoch {i+1}: {T} iterations \n-------------------------------")
    t0 = max_iter
    flag = True

    i = 0  # global iteration counter
    min_T = T**0.5
    # measure running time
    start = time.perf_counter()
    while not optimizer.complete and i < max_iter:
        prev_noisy_loss = None
        rho_0 = rho / (T + 1)
        sigma_f = sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / 4)) ** 0.5
        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        if T < min_T:
            max_T = T
            flag = True
        else:
            max_T = T // 4
            flag = False
        for i in range(min(max_T, max_iter - i)):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)

            # TODO: update rho (privacy budget) correctly
            XX, yy = X, y

            # Compute prediction error
            loss = model(XX, yy)
            noisy_loss = (
                loss + rng.normal(0, sigma_f * opt_params["f_sensitivity"])
            ).item()
            if i % print_every == 0:
                print(f"Iteration {i}: {loss.item():>.5f}")
                # if i < t0:
                # check if the loss is increasing to previous iteration
                # TODO: consider setting a more reasonable threshold using MIN_DECREASE
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            hess_closure = lambda: torch.autograd.functional.hessian(
                lambda w: model(XX, yy, w), model.w
            )
            optimizer.step(lambda: model(XX, yy), hess_closure)
            wandb.log({"loss": loss})
            wandb.log({"grad_norm": model.w.grad.norm()})

            if optimizer.complete:
                print("Optimization complete!")
                break
            if (
                prev_noisy_loss is not None
                and noisy_loss > prev_noisy_loss + 4 * sigma_f_unscaled
            ):
                print(f"With T={T}, loss is increasing. Shrink T to {T // 4}")

                rho_prev = rho
                rho *= 1 - (i + 1) / T
                T //= 4
                # print change of rho
                print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
                break
            else:
                prev_noisy_loss = noisy_loss
        else:
            if flag:
                rho = 0
                break
            print(f"1/4 of the budget used. Shrink T={T} to {T // 4}")
            rho_prev = rho
            rho *= 1 - (i + 1) / T
            print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
            T //= 4
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")
    # print(batch_frac)
    # mech_t = optimizer.compose_rdp(batch_frac)
    # mechanism_lst.append(mech_t)
    # mech = mech_t if i == 0 else compose([mech, mech_t], [1, 1])
    # train_loss = full_loss_closure()
    # print("loss: {loss:>5f}, loss on the whole dataset: {train_loss:>5f}")
    # loss, loss.item()
    # if eval_on_train:
    #     train_loss = full_loss_closure()
    #     print(
    #         f"batch size = {batch_size}, loss: {loss:>5f}, loss on the whole dataset: {train_loss:>5f}"
    #     )
    # else:
    #     print(f"batch size = {batch_size}, loss: {loss:>5f}")
    # print final loss
    grad_norm = model.w.grad.norm()
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, privacy budget left: {rho:.5f}"
    )
    print(optimizer.grad_evals, optimizer.hess_evals)
    # store loss, grad_norm, runtime in dictionary
    results = dict(
        loss=loss, grad_norm=grad_norm, runtime=runtime, rho_left=rho, num_iter=i
    )
    wandb.run.summary.update(results)
    wandb.finish()
    return model, optimizer, results


def tr_exp(
    alpha,
    G,
    M,
    initial_gap,
    rho=1,
    print_every=50,
    seed=22,
    wandb_on=True,
):
    T = int(np.ceil(6 * M**0.5 * initial_gap / alpha**1.5))
    wandb.init(
        project=WANDB_PROJECT,
        group=f"eps_g={eps_g_target},seed={seed}",
        config={
            "method": "dptr",
            "seed": seed,
            "rho": rho,
            "eps": rdp2dp(rho, 1 / n),
            "eps_g": opt_params["eps_g"],
            "eps_H": opt_params["eps_H"],
            "max_iter": T,
        },
        mode="online" if wandb_on else "disabled",
    )
    torch.manual_seed(seed)
    rng = default_rng(seed)
    model = ERM(n_dim, regularizer)
    # full_loss_closure = lambda: model(X, y).item()
    print(f"Optimize for at most T={T} iterations")

    opt_tr_params = {
        "alpha": alpha,
        "G": G,
        "M": M,
        "n": n,
        "n_dim": n_dim,
        "grad_sensitivity": G,
        "hess_sensitivity": M,
        "T": T,
    }
    print(opt_tr_params)
    optimizer = DPTR(model.parameters(), opt_tr_params, rho=rho)
    # batch_size = 5000
    # batch_frac = batch_size / n
    start = time.perf_counter()
    for i in range(T):
        # indices = rng.choice(n, size=batch_size)
        # XX, yy = X[indices], y[indices]
        # XX, yy = XX.to(device), yy.to(device)
        XX, yy = X, y

        # Compute prediction error
        loss = model(XX, yy)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        hess_closure = lambda: torch.autograd.functional.hessian(
            lambda w: model(XX, yy, w), model.w
        )
        optimizer.step(lambda: model(XX, yy), hess_closure)
        wandb.log({"loss": loss})
        wandb.log({"grad_norm": model.w.grad.norm()})

        if optimizer.complete:
            print("Optimization complete!")
            break
        if i % print_every == 0:
            print(f"Iteration {i}: {loss.item():.5f}")
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")
    grad_norm = model.w.grad.norm()
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, privacy budget left: {rho:.5f}"
    )
    results = dict(
        loss=loss,
        grad_norm=grad_norm,
        runtime=runtime,
        rho_left=rho * i / T,
        num_iter=i,
    )
    wandb.run.summary.update(results)
    wandb.finish()
    return model, optimizer, results


# Note that the DPOPT algorithm outputs a ((1+c1)eps_g, (1+c)eps_H)-approximate second-order necessary point.
c1, c = 0.3, 1 / 12
# eps_g, eps_H = 0.1 / (1 + c1), 0.316 / (1 + c)
eps_g_target = 0.01
eps_g, eps_H = eps_g_target / (1 + c1), eps_g_target**0.5 / (1 + c)
# eps_g, eps_H = 0.0001, 0.01
loss_sensitivity, G, M = 1, 1, 1
b_g, b_H = 10, 10

# fmt: off
opt_params = dict(
    eps_g=eps_g, eps_H=eps_H, n_dim=n_dim, n=n, loss_sensitivity=loss_sensitivity, B_g=1, B_H=1, G=G, M=M, lower_bound=0.3, c=c, c1=c1, c2=1 / 12, b_g=b_g, beta_g=0.6, c_g=0.6, b_H=b_H, beta_H=0.2, c_H=0.25, checked=False,
)

check_and_compute_params(opt_params)
print_every = 10
initial_gap = 0.5
def dp2rdp(eps, delta):
    i = np.log(1/delta)
    return (np.sqrt(eps + i) - np.sqrt(i)) ** 2

def rdp2dp(rho, delta):
    # reverse
    i = np.log(1/delta)
    return (rho ** 0.5 + np.sqrt(i)) ** 2 - i

eps_lst = np.arange(0.1, 2.1, 0.2)
# eps_lst = np.array([0.1, 0.5, 1.0])
rhos = dp2rdp(eps_lst, 1 / n)
# rho2rdp_map = dict(zip(rhos, eps_lst))
# print(f"eps={eps:.2f}, rho={rho:.5f}")

ls = False
wandb_on = True
SEED = 21

def exp_range_eps():
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    output_file = open(os.path.join("results", f"result-eps_g={eps_g:.4f}_{time_str}.csv"), 'w')
    output_file.write("method,eps,rho,loss,grad_norm,runtime,rho_left,num_iter\n")

    for eps, rho in zip(eps_lst, rhos):
        for ls in [True, False]:
            method = "DPOPT-LS" if ls else "DPOPT"
            print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
            model, optimizer, results = dpopt_exp(opt_params, initial_gap=initial_gap, rho=rho, line_search=ls, max_iter=2000, init_T=10000, print_every=print_every, seed=SEED, wandb_on=wandb_on)
            print("")
            print("<<<<<")
            # write result to file
            output_file.write(f"{method},{eps:.4f},{rho:.5f},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f},{results['rho_left']:.5f},{results['num_iter']}\n")
    # model, optimizer = tr_exp(alpha=eps_g, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
    print("Running DPTR...")
    for eps, rho in zip(eps_lst, rhos):
        method = "DPTR"
        print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
        model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
        print("")
        print("<<<<<")
        # write result to file
        output_file.write(f"{method},{eps:.4f},{rho:.5f},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f}, {results['rho_left']:.5f},{results['num_iter']}\n")

    output_file.close()

exp_range_eps()
# %%
# rho = .1
# model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
