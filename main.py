from datetime import datetime
import os
import time

import warnings
import scipy.linalg
import numpy as np
import pandas as pd
from torch.distributions.laplace import Laplace
from numpy.random import default_rng
from icecream import ic
import torch

from DPTR import DPTR
from DPOPT import DPOPT

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

# rho2rdp_map = dict(zip(rhos, eps_lst))
# print(f"eps={eps:.2f}, rho={rho:.5f}")

def exp_range_eps(eps_lst, rhos=None, wandb_on=True, seed=21):
    if rhos is None:
        rhos = dp2rdp(eps_lst, 1 / n)
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    output_file = open(os.path.join("results", f"result-eps_g={eps_g:.4f}_{time_str}.csv"), 'w')
    output_file.write("method,eps,rho,loss,grad_norm,runtime,rho_left,num_iter\n")

    for eps, rho in zip(eps_lst, rhos):
        for ls in [True, False]:
            method = "DPOPT-LS" if ls else "DPOPT"
            print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
            model, optimizer, results = dpopt_exp(opt_params, initial_gap=initial_gap, rho=rho, line_search=ls, max_iter=2000, init_T=10000, print_every=print_every, seed=seed, wandb_on=wandb_on)
            print("")
            print("<<<<<")
            # write result to file
            output_file.write(f"{method},{eps:.4f},{rho:.5f},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f},{results['rho_left']:.5f},{results['num_iter']}\n")
    # model, optimizer = tr_exp(alpha=eps_g, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
    print("Running DPTR...")
    for eps, rho in zip(eps_lst, rhos):
        method = "DPTR"
        print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
        model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, seed=seed, wandb_on=wandb_on)
        print("")
        print("<<<<<")
        # write result to file
        output_file.write(f"{method},{eps:.4f},{rho:.5f},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f}, {results['rho_left']:.5f},{results['num_iter']}\n")

    output_file.close()

eps_lst = np.arange(0.1, 0.2, 0.2)
# eps_lst = np.array([0.1, 0.5, 1.0])
rhos = dp2rdp(eps_lst, 1 / n)
wandb_on = False
SEED = 21

# exp_range_eps(eps_lst, rhos, wandb_on=wandb_on, seed=SEED)
# %%
rho = .1
model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
