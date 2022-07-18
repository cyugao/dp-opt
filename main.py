from datetime import datetime
import os
import time

import numpy as np
import pandas as pd
from numpy.random import default_rng
from icecream import ic
import torch

from DPTR import DPTR
from DPOPT import DPOPT

import wandb

WANDB_PROJECT = "dpopt"

# %%

df_covtype = pd.read_csv("data/covtype.csv")
df_covtype.head()
normalize_df = lambda df: (df - df.mean()) / df.std()
# Standardize numerical columns
df_covtype.iloc[:, :10] = normalize_df(df_covtype.iloc[:, :10])


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


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def regularizer(w):
    w2 = w * w
    return (w2 / (1 + w2)).sum()


# %%


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

    delta = 0.25 * (1 - c - c_H) ** 2 - 2 / 3 * c2
    assert delta > 0
    opt_params["t2"] = 1.5 * (1 - c - c_H) + 3 * delta**0.5

    gamma_g_bar = opt_params["gamma_g_bar"]
    t2 = opt_params["t2"]
    term1_ls = 0.5 * c_g * gamma_g_bar * eps_g * eps_g
    term2_ls = 0.25 * c_H * t2 * t2 * eps_H / (M * M)
    opt_params["min_decrease_ls"] = min(term1_ls, term2_ls)

    assert c1 < 1 / 2
    assert c2 + c < 1 / 3
    term1 = (1 - 2 * c1) / (2 * G) * (eps_g * eps_g)
    term2 = 2 * (1 / 3 - c2 - c) * eps_H**3 / (M * M)
    opt_params["min_decrease"] = min(term1, term2)

    opt_params["checked"] = True


def dp_estimate_T(gap, line_search=True):
    assert opt_params["checked"]
    if line_search:
        decrease = opt_params["min_decrease_ls"]
    else:
        decrease = opt_params["min_decrease"]

    return int(np.ceil(gap / decrease))


def one_step(model, optimizer, X, y):
    loss = model(X, y)
    optimizer.zero_grad()
    loss.backward()
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(X, y, w), model.w
    )
    optimizer.step(lambda: model(X, y), hess_closure)
    wandb.log({"loss": loss})
    wandb.log({"grad_norm": model.w.grad.norm()})
    return loss


def opt_shrink_T(model, optimizer, rho, init_T, min_T, max_iter, print_every):
    i = 0  # global iteration counter
    T = init_T
    while not optimizer.complete and i < max_iter:
        prev_noisy_loss = None
        rho_0 = rho / (T + 1)
        sigma_f = sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / 4)) ** 0.5
        sigma_f_unscaled = sigma_f * opt_params["f_sensitivity"]
        # print(sigma_f_unscaled)
        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        if T < min_T:
            TT = T
            stop = True
        else:
            TT = T // 4  # only run up to 1/4 * T to speed up
            stop = False
        for j in range(min(TT, max_iter - i)):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)

            loss = one_step(model, optimizer, X, y)
            noisy_loss = (loss + sigma_f_unscaled * torch.randn()).item()

            if i % print_every == 0:
                print(f"Iteration {i}: {loss.item():>.5f}")

            if optimizer.complete:
                print("Optimization complete!")
                break
            if (
                prev_noisy_loss is not None
                and noisy_loss > prev_noisy_loss + 2 * sigma_f_unscaled
                # randomness here (different from the constant used in opt_expand_T)
            ):
                print(
                    f"With T={T}, loss is increasing or decreasing slowly. Shrink T to {T // 4}"
                )
                break
            else:
                prev_noisy_loss = noisy_loss
        else:
            if stop:
                print("All iterations completed!")
                rho = 0
                break
            print(f"1/4 of the budget used. Shrink T={T} to {T // 4}")
        rho_prev = rho
        rho *= 1 - (j + 1) / T
        T //= 4
        print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
    return loss, rho, i


def opt_grow_T(
    model,
    optimizer,
    rho,
    init_T,
    min_T,
    max_iter,
    print_every,
    fallback_rho_frac=0.25,
):
    i = 0  # global iteration counter
    T = min_T
    fallback_rho = rho * fallback_rho_frac
    rho -= fallback_rho
    while not optimizer.complete and rho > 0:
        if T >= init_T:
            T = init_T
            break
        prev_noisy_loss = None
        rho_0 = rho / T
        sigma_f = sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / 4)) ** 0.5
        sigma_f_unscaled = sigma_f * opt_params["f_sensitivity"]
        # print(sigma_f_unscaled)
        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        for j in range(min(T, max_iter - i)):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)
            loss = one_step(model, optimizer, X, y)
            if i % print_every == 0:
                print(f"Iteration {i}: {loss.item():>.5f}")
            noisy_loss = (loss + sigma_f_unscaled * torch.randn()).item()

            if optimizer.complete:
                print("Optimization complete!")
                break
            if (
                prev_noisy_loss is not None
                and noisy_loss > prev_noisy_loss + 4 * sigma_f_unscaled
                # the constant 4 here accounts for more conservative checking here
            ):
                print(f"With T={T}, loss is increasing. Grow T to {T * 2}")
                break
            else:
                prev_noisy_loss = noisy_loss
        rho_prev = rho
        rho *= 1 - (j + 1) / T
        T *= 2
        print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")

    rho += fallback_rho
    if not optimizer.complete:
        rho_0 = rho / T
        sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / 3)) ** 0.5
        # print(sigma_f_unscaled)
        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        for j in range(min(T, max_iter - i)):
            i += 1
            loss = one_step(model, optimizer, X, y)

            if i % print_every == 0:
                print(f"Iteration {i}: {loss.item():>.5f}")

            if optimizer.complete:
                print("Optimization complete!")
                break
        rho_prev = rho
        rho *= 1 - (j + 1) / T
        print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
    return loss, rho, i


def opt_adapt_T(
    model,
    optimizer,
    rho,
    init_T,
    min_T,
    max_iter,
    print_every,
    estimate_T_closure,
    update_T_every=20,
):
    i = 0  # global iteration counter
    T = init_T
    while not optimizer.complete and i < max_iter:
        rho_0 = rho / (T + 1)
        sigma_f = (1 / (2 * rho_0)) ** 0.5  # Budget for updating the estimate of T
        sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / 3)) ** 0.5
        optimizer.new_epoch(sigma_g, sigma_H, lambda_svt)
        steps = update_T_every if T > min_T else min_T
        for j in range(steps):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)
            loss = one_step(model, optimizer, X, y)
            if i % print_every == 0:
                print(f"Iteration {i}: {loss.item():>.5f}")

            if optimizer.complete:
                print("Optimization complete!")
                rho *= 1 - (j + 1) / (T + 1)
                break
        else:
            if T == min_T:
                print(
                    "This should not happen since optimzation should be completed by now!"
                )
            last_T = T
            T = estimate_T_closure(sigma_f)
            # print T change
            print(f"Update T: {last_T} -> {T}")
            rho *= 1 - (update_T_every + 1) / (T + 1)

    return loss, rho, i


def dpopt_exp(
    opt_params,
    strategy,
    initial_gap=None,
    rho=1,
    max_iter=1000,
    line_search=True,
    print_every=50,
    seed=22,
    wandb_on=True,
):
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
    if initial_gap:
        gap = initial_gap
    else:
        rho_f = rho / max_iter  # TODO: just a simple heuristic for now
        rho -= rho_f
        sigma_f = (1 / (2 * rho_f)) ** 0.5
        gap = full_loss_closure() - opt_params["lower_bound"]
        gap += sigma_f * opt_params["f_sensitivity"] * abs(rng.normal(1))
    init_T = dp_estimate_T(gap, line_search=line_search)
    # print(f"Estimated T={init_T}")
    min_T = int(init_T**0.5)  # TODO: another heuristic
    # measure running time
    start = time.perf_counter()
    if strategy == "adaptive":

        def estimate_T_closure(sigma_f):
            gap = full_loss_closure() - opt_params["lower_bound"]
            gap += sigma_f * opt_params["f_sensitivity"] * abs(rng.normal(1))
            return dp_estimate_T(gap, line_search=line_search)

        out = opt_adapt_T(
            model,
            optimizer,
            rho,
            init_T,
            min_T,
            max_iter,
            print_every,
            estimate_T_closure,
        )
    elif strategy == "shrink":
        out = opt_shrink_T(model, optimizer, rho, init_T, min_T, max_iter, print_every)
    elif strategy == "grow":
        out = opt_grow_T(model, optimizer, rho, init_T, min_T, max_iter, print_every)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    loss, rho_left, num_iter = out
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")
    grad_norm = model.w.grad.norm()
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, privacy budget left: {rho:.5f}"
    )
    print(optimizer.grad_evals, optimizer.hess_evals)
    # store loss, grad_norm, runtime in dictionary
    results = dict(
        loss=loss,
        grad_norm=grad_norm,
        runtime=runtime,
        rho_left=rho_left,
        num_iter=num_iter,
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
        loss = one_step(model, optimizer, X, y)

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
    output_file.write("method,eps,rho,complete,loss,grad_norm,runtime,rho_left,num_iter\n")

    for eps, rho in zip(eps_lst, rhos):
        for ls in [True, False]:
            method = "DPOPT-LS" if ls else "DPOPT"
            print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
            model, optimizer, results = dpopt_exp(opt_params, initial_gap=initial_gap, rho=rho, line_search=ls, max_iter=2000, print_every=print_every, seed=seed, wandb_on=wandb_on)
            print("")
            print("<<<<<")
            # write result to file
            output_file.write(f"{method},{eps:.4f},{rho:.5f},{optimizer.complete},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f},{results['rho_left']:.5f},{results['num_iter']}\n")
    # model, optimizer = tr_exp(alpha=eps_g, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
    print("Running DPTR...")
    for eps, rho in zip(eps_lst, rhos):
        method = "DPTR"
        print(f">>>>>\n Running {method} with rho={rho:.5f} (eps={eps:.2f})")
        model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, seed=seed, wandb_on=wandb_on)
        print("")
        print("<<<<<")
        # write result to file
        output_file.write(f"{method},{eps:.4f},{rho:.5f},{optimizer.complete},{results['loss']:.5f},{results['grad_norm']:.5f},{results['runtime']:.5f}, {results['rho_left']:.5f},{results['num_iter']}\n")

    output_file.close()

eps_lst = np.arange(0.1, 0.2, 0.2)
# eps_lst = np.array([0.1, 0.5, 1.0])
rhos = dp2rdp(eps_lst, 1 / n)
wandb_on = False
SEED = 21

# exp_range_eps(eps_lst, rhos, wandb_on=wandb_on, seed=SEED)
rho = 0.1
strategy = "grow"
dpopt_exp(opt_params, strategy, initial_gap=initial_gap, rho=rho, line_search=True, max_iter=2000, print_every=print_every, seed=SEED, wandb_on=wandb_on)
model, optimizer, results = tr_exp(alpha=eps_g_target, G=1, M=1, initial_gap=initial_gap, rho=rho, print_every=print_every, wandb_on=wandb_on)
