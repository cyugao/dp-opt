from datetime import datetime
import argparse
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numpy.random import default_rng
from icecream import ic
import torch
torch.set_num_threads(28)

from DPGD import DPGD
from DPTR import DPTR
from DPOPT import DPOPT
from DPOPT import GRADIENT_STEP, CURVATURE_STEP
from timeout import timeout
from timeout import TimeoutError

import wandb

def parse_args():
    # parse dataset, loss_fn, regularizer, eps_g_target and seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="covtype")
    parser.add_argument("--loss_fn", type=str, default="sigmoid")
    parser.add_argument("--regularizer", "-reg", type=str, default="l2")
    parser.add_argument("--eps_g", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    # parse run method
    parser.add_argument("--method", type=str, default="all")
    args = parser.parse_args()
    return args

args = parse_args()

WANDB_PROJECT = f"{args.dataset}-{args.loss_fn}"
# WANDB_PROJECT = "test"

# %%

normalize_df = lambda df: (df - df.mean()) / df.std()

def load_covtype():
    df = pd.read_csv("data/covtype.csv")
    # Standardize numerical columns
    df.iloc[:, :10] = normalize_df(df.iloc[:, :10])

    # only keep cover type 1 and 2 for binary classification
    df = df[df["Cover_Type"] <= 2]

    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    # Change labels to +1 and -1
    y[y == 2] = -1
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    # expected norm sqrt(10+1+1) = 3.464
    X /= 4
    return X, y


def load_ijcnn():
    df = pd.read_csv("data/ijcnn.csv")
    # Standardize numerical columns
    df.iloc[:, 10:] = normalize_df(df.iloc[:, 10:])

    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    X /= 3
    return X, y


# def load_mnist():
#     from torchvision import datasets, transforms

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     train_set = datasets.MNIST(
#         "./data/mnist/train",
#         train=True,
#         transform=transform,
#         download=True,
#     )
#     # test_set = datasets.MNIST(
#     #     "./data/mnist/test", train=False, transform=transforms.ToTensor(), download=True
#     # )
#     return train_set.data, train_set.targets

def load_mnist():
    # load Tensor and apply transform from data/mnist.npz
    data = np.load("data/mnist.npz")
    X = torch.Tensor(data["x_train"])
    y = torch.Tensor(data["y_train"])
    mean, std = 0.1307, 0.3081
    X = (X / 255 - mean) / (std * 33)
    X = X.flatten(start_dim=1)
    y[y < 5] = -1
    y[y >= 5] = 1
    return X, y
    

# dataset = "covtype"
if args.dataset == "covtype":
    X, y = load_covtype()
elif args.dataset == "ijcnn":
    X, y = load_ijcnn()
elif args.dataset == "mnist":
    X, y = load_mnist()
print(f"Using dataset {args.dataset}, size: {X.shape}, target eps_g={args.eps_g:.2f}", )
print()
# train test split
# rng = default_rng(0)
# indices = rng.choice(len(X), size=50000)
# X, y = X[indices], y[indices]

n, n_dim = X.shape
X.shape, y.shape


class ERM(torch.nn.Module):
    def __init__(self, input_dim, loss_fn, regularizer, reg_coef=1e-3):
        super(ERM, self).__init__()
        self.input_dim = input_dim
        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.reg_coef = reg_coef

        self.w = torch.nn.Parameter(torch.randn(self.input_dim))

    def forward(self, X, y, w=None):
        if w is None:
            w = self.w
        t = y * (X @ w)
        return self.loss_fn(t).mean() + self.reg_coef * self.regularizer(w)


# class ERM_Multi(torch.nn.Module):
#     def __init__(self, input_dim, regularizer, n_class=10, reg_coef=1e-3):
#         super(ERM_Multi, self).__init__()
#         self.input_dim = input_dim
#         self.regularizer = regularizer
#         self.reg_coef = reg_coef

#         self.dense = torch.nn.Linear(self.input_dim, n_class)

#     def forward(self, X, y, w=None):
#         + self.reg_coef * self.regularizer(w)


def tukey_regularizer(w):
    w2 = w * w
    return (w2 / (1 + w2)).sum()

def l2_regularizer(w):
    return .5 * w @ w

if args.regularizer == "l2":
    regularizer = l2_regularizer
elif args.regularizer == "tukey":
    regularizer = tukey_regularizer

if args.loss_fn == "sigmoid":
    loss_fn = torch.sigmoid
elif args.loss_fn == "logistic":
    loss_fn = lambda t: torch.log1p(exp(-t))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# %%
def dp2rdp(eps, delta):
    i = np.log(1 / delta)
    return (np.sqrt(eps + i) - np.sqrt(i)) ** 2


def rdp2dp(rho, delta):
    # reverse
    i = np.log(1 / delta)
    return (rho**0.5 + np.sqrt(i)) ** 2 - i


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

    assert c1 + c_g < 1
    opt_params["gamma_g_bar"] = 2 * (1 - c1 - c_g) / G

    delta = 0.25 * (1 - c - c_H) ** 2 - 2 / 3 * c2
    assert delta > 0
    opt_params["t1"] = 1.5 * (1 - c - c_H) - 3 * delta**0.5
    opt_params["t2"] = 1.5 * (1 - c - c_H) + 3 * delta**0.5
    # print(opt_params["t1"] / opt_params["t2"]) => 0.17

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


def dp_estimate_T(opt_params, gap, line_search=True):
    assert opt_params["checked"]
    if line_search:
        decrease = opt_params["min_decrease_ls"]
    else:
        decrease = opt_params["min_decrease"]

    return int(np.ceil(gap / decrease))


def one_step(model, optimizer, logging=True):
    if batch_size:
        assert rng is not None
        indices = rng.choice(n, size=batch_size)
        XX, yy = X[indices], y[indices]
        XX, yy = XX.to(device), yy.to(device)
    else:
        XX, yy = X, y
    optimizer.zero_grad()
    loss = model(XX, yy)
    loss.backward()
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(XX, yy, w), model.w
    )
    optimizer.step(lambda: model(XX, yy), hess_closure)
    grad_norm = model.w.grad.norm()
    if logging:
        wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item()})
    return loss


def one_step_extra(model, optimizer, X=X, y=y, logging=True):
    optimizer.zero_grad()
    loss = model(X, y)
    loss.backward()
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(X, y, w), model.w
    )
    STEP, noisy_value = optimizer.step(lambda: model(X, y), hess_closure)
    grad_norm = model.w.grad.norm()
    if logging:
        wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item()})
    return loss, (STEP, noisy_value)


def opt_shrink_T(
    model, optimizer, rho, init_T, min_T, max_iter, print_every, opt_params
):
    i = 0  # global iteration counter
    T = init_T
    while not optimizer.completed and i < max_iter:
        prev_noisy_loss = None
        rho_0 = rho / (T + 1)
        sigma_f = sigma_g = sigma_H = lambda_svt = (1 / (2 * rho_0 / (3 + optimizer.line_search))) ** 0.5
        if not optimizer.line_search:
            lambda_svt = None
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

            loss = one_step(model, optimizer)
            noisy_loss = (loss + sigma_f_unscaled * torch.randn(1)).item()

            if i % print_every == 0:
                print(f"Iteration {i:>5d}: {loss.item():>.5f}")

            if optimizer.completed:
                print("\nOptimization complete!")
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


def opt_two_phase_T(
    model,
    optimizer,
    rho,
    init_T,
    max_iter,
    print_every,
    fallback_rho_frac=0.25,
):
    T = int(init_T ** 0.5) # Just a heuristic
    fallback_rho = rho * fallback_rho_frac
    rho -= fallback_rho
    rho_0 = rho / T
    if batch_size:
        subsample_prob = batch_size / n
        eps = rdp2dp(rho, 1 / n)
        mech = calibrate_rho(eps, delta, T, subsample_prob, optimizer.line_search)
        rho_0 = mech.params["rho"] / T
        print(f"With subsample prob {subsample_prob:.4f}, rho={mech.params['rho']:.5f}")
        print(f"Without subsample prob, rho={rho:.5f}")

    optimizer.new_epoch_rho(rho_0)
    for i in range(T):
        loss, (STEP, noisy_value) = one_step_extra(model, optimizer)
        if i % print_every == 0:
            step_str = "grad_norm" if STEP == GRADIENT_STEP else "hess_eig"
            print(f"Iteration {i:>5d}: {'mini-batch ' if batch_size else ''}loss={loss.item():>.5f}, {step_str}={noisy_value:.5f}")
        if optimizer.completed:
            print("\nOptimization complete!")
            break
    if batch_size is None:
        rho *= 1 - (i + 1) / T
    else:
        mech_run = create_complex_mech(mech.params["rho"], T, subsample_prob, line_search=optimizer.line_search, evals=[optimizer.grad_evals, optimizer.hess_evals])
        rho -= dp2rdp(mech_run.get_approxDP(delta), delta)
        # input(f"rho={rho:.5f}, OK!")

    rho += fallback_rho
    T = init_T
    if not optimizer.completed:
        print(f"Entering fallback: remaining privacy budget {rho:.5f}\n>>>>>")
        rho_0 = rho / T
        if batch_size:
            eps = rdp2dp(rho, 1 / n)
            mech = calibrate_rho(eps, delta, T, subsample_prob, optimizer.line_search)
            rho_0 = mech.params["rho"] / T
            print(f"With subsample prob {subsample_prob:.4f}, rho={mech.params['rho']:.5f}")
            print(f"Without subsample prob, rho={rho:.5f}")
        optimizer.new_epoch_rho(rho_0)
        cur_loss = 100
        inc_loss_count = 0
        for j in range(min(T, max_iter - i)):
            i += 1
            loss, (STEP, noisy_value) = one_step_extra(model, optimizer)
            if i % print_every == 0:
                step_str = "grad_norm" if STEP == GRADIENT_STEP else "hess_eig"
                print(f"Iteration {i:>5d}: {'mini-batch ' if batch_size else ''}loss={loss.item():>.5f}, {step_str}={noisy_value:.5f}")
            if i % 10 == 0 and batch_size is None:
            # a heuristic to stop early (n is not large enough)
            # mini-batch loss is not guaranteed to decrease, so we only apply to the full batch case
                last_loss = cur_loss
                cur_loss = loss.item()
                if cur_loss > last_loss:
                    inc_loss_count += 1
                    if inc_loss_count >= 3:
                        print("Loss is increasing. Stopping...")
                        break
                else:
                    inc_loss_count = 0

            if optimizer.completed:
                print("\nOptimization complete!")
                break
        rho_prev = rho
        if batch_size is None:
            rho *= 1 - (i + 1) / T
        else:
            mech_run = create_complex_mech(mech.params["rho"], T, subsample_prob, line_search=optimizer.line_search, evals=[optimizer.grad_evals, optimizer.hess_evals])
            rho -= dp2rdp(mech_run.get_approxDP(delta), delta)
        print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
    return loss, rho, i


def opt_fixed_T(model, optimizer, rho, init_T, print_every):
    T = init_T
    if batch_size:
        subsample_prob = batch_size / n
        eps = rdp2dp(rho, 1 / n)
        mech = calibrate_rho(eps, delta, T, subsample_prob, optimizer.line_search)
        rho_0 = mech.params["rho"] / T
        print(f"With subsample prob {subsample_prob:.4f}, rho={mech.params['rho']:.5f}")
        print(f"Without subsample prob, rho={rho:.5f}")
    else:
        rho_0 = rho / T
    optimizer.new_epoch_rho(rho_0)
    # Running main loop
    print("\nRunning main loop...")
    cur_loss = 100
    inc_loss_count = 0
    for i in range(T):
        loss, (STEP, noisy_value) = one_step_extra(model, optimizer)
        if i % print_every == 0:
            step_str = "grad_norm" if STEP == GRADIENT_STEP else "hess_eig"
            print(f"Iteration {i:>5d}: {'mini-batch ' if batch_size else ''}loss={loss.item():>.5f}, {step_str}={noisy_value:.5f}")

        if i % 10 == 0 and batch_size is None:
            # a heuristic to stop early (n is not large enough)
            # mini-batch loss is not guaranteed to decrease, so we only apply to the full batch case
            last_loss = cur_loss
            cur_loss = loss.item()
            if cur_loss > last_loss:
                inc_loss_count += 1
                if inc_loss_count >= 3:
                    print("Loss is increasing. Stopping...")
                    break
            else:
                inc_loss_count = 0

        if optimizer.completed:
            print("\nOptimization complete!")
            break
    if batch_size is None:
        rho *= 1 - (i + 1) / T
    else:
        mech_run = create_complex_mech(mech.params["rho"], T, subsample_prob, line_search=optimizer.line_search, evals=[optimizer.grad_evals, optimizer.hess_evals])
        rho -= dp2rdp(mech_run.get_approxDP(delta), delta)
    return loss, rho, i


# %%
from autodp.mechanism_zoo import ExactGaussianMechanism, PureDP_Mechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling

subsample = AmplificationBySampling(PoissonSampling=False)


def create_complex_mech(rho, T, subsample_prob, line_search, num_trials=1, improved_bound_flag=True, evals=None):
    sigma = ((2 + int(line_search))* T / (2 * rho)) ** 0.5
    eps = 1 / sigma
    gm = ExactGaussianMechanism(sigma, name="GM")
    svt = PureDP_Mechanism(eps=eps, name="SVT")
    gm.neighboring = "replace_one"
    svt.neighboring = "replace_one"

    compose = Composition()
    sample_with_prob = lambda mech: subsample(mech, subsample_prob, improved_bound_flag=improved_bound_flag)
    sample_gm = sample_with_prob(gm)

    if evals is None:
        # (grad_evals, hess_evals)
        evals = [T, T]
    
    if evals[1] == 0:
        # no hessian evals
        mech_lst = [sample_gm]
        evals = [evals[0]]
    else:
        mech_lst = [sample_gm, sample_gm]
    if line_search:
        mech_lst.append(sample_with_prob(svt))
        evals.append(T)
    mech = compose(mech_lst, evals)
    if num_trials > 1:
        return compose([mech], [num_trials])
    return mech


# next we can create it as a mechanism class, which requires us to inherit the base mechanism class,
#  which we import now
from autodp.autodp_core import Mechanism
from autodp.calibrator_zoo import generalized_eps_delta_calibrator


class Complex_Mechanism(Mechanism):
    def __init__(self, params, name="A_Good_Name"):
        self.name = name
        self.params = params
        mech = create_complex_mech(
            params["rho"],
            params["T"],
            params["subsample_prob"],
            line_search=params["line_search"],
            num_trials=params["num_trials"],
        )
        # The following will set the function representation of the complex mechanism
        # to be the same as that of the mech
        self.set_all_representation(mech)


general_calibrate = generalized_eps_delta_calibrator()


@timeout(seconds=120)
def calibrate_rho(eps_budget, delta, T, subsample_prob, line_search, num_trials=1, bounds=(1e-7, 1)):
    params = {
        "rho": None,
        "T": T,
        "subsample_prob": subsample_prob,
        "num_trials": num_trials,
        "line_search": line_search,
    }
    rho_lo, rho_up = bounds
    params["rho"] = rho_up
    mech = Complex_Mechanism(params)
    if mech.get_approxDP(delta) < eps_budget:
        # rho can be very large
        return mech
    
    params["rho"] = rho_lo
    mech = Complex_Mechanism(params)
    if mech.get_approxDP(delta) > eps_budget:
        raise ValueError("privacy budget is not enough")

    return general_calibrate(
        Complex_Mechanism,
        eps_budget,
        delta,
        bounds,
        params=params,
        para_name="rho",
        name="Complex_Mechanism",
    )


def estimate_lower_bound(
    model,
    optimizer,
    eps_total,
    delta_total,
    T,
    print_every,
    batch_size=5000,
    num_trials=5,
):
    """
    Estimate the lower bound of f using specified privacy budget.
    Averaged over {num_trials} times"""
    eps, delta = eps_total / num_trials, delta_total / num_trials
    subsample_prob = batch_size / n
    ic(eps, delta, T, subsample_prob)
    mech = calibrate_rho(eps, delta, T, subsample_prob, optimizer.line_search, num_trials)
    rho = mech.params["rho"]
    rho_0 = rho / T

    optimizer.new_epoch_rho(rho_0)

    lower_bounds = np.zeros(num_trials)
    rho_left = 0
    for i in range(num_trials):
        indices = rng.choice(n, size=batch_size)
        XX, yy = X[indices], y[indices]
        # for j in range(T):
        #     loss = one_step(model, optimizer, XX, yy)
        #     if j % print_every == 0:
        #         print(
        #             f"(Trial {i}) Iteration {j}: loss={loss.item():>.5f}, grad_norm={model.w.grad.norm():.5f}"
        #         )
        #     if optimizer.completed:
        #         # rho_left += rho_0 * (T - j - 1) / T
        #         print(f"Trial {i} complete.")
        #         optimizer.completed = False
        #         break
        min_T = 20
        max_iter = 500
        #TODO: change to opt_fixed_T
        loss, rho, _ = opt_adapt_noise(model, optimizer, rho, T, min_T, max_iter, print_every, XX, yy, logging=False)
        lower_bounds[i] = loss
        rho_left += rho
        optimizer.completed = False
    return lower_bounds, rho_left


def opt_adapt_T(
    model,
    optimizer,
    rho,
    init_T,
    min_T,
    max_iter,
    print_every,
    estimate_T_closure,
    update_T_every=100,
    estimate_lb=False,
):
    if estimate_lb:
        try:
            # estimate lower bound of f
            delta = 1 / n
            eps = rdp2dp(rho, delta)
            eps1, delta1 = eps / 4, delta / 4
            print("Estimating the lower bound of f...")
            eps_g, eps_H = optimizer.eps_g, optimizer.eps_H

            optimizer.set_sosp_params(eps_g * 10, eps_H * 10 ** 0.5)
            lower_bounds, rho_left = estimate_lower_bound(
                model,
                optimizer,
                eps1,
                delta1,
                init_T,
                print_every,
                batch_size=5000,
                num_trials=5,
            )
            optimizer.set_sosp_params(eps_g, eps_H)
            lb_mean, lb_std = lower_bounds.mean(), lower_bounds.std()
            print(
                f"Estimated lower bound: {lb_mean:.5f}, standard deviation: {lb_std:.5f}\n"
            )
            opt_params["lower_bound"] = (
                lb_mean - 2 * 1.414 * lb_std
            )  # 95% confidence interval lower end        # can estimate variance by doing this multiple times
            rho = dp2rdp(eps - eps1, delta - delta1) + rho_left
        except TimeoutError:
            print("TimeoutError: estimating lower bound of f")
    # T = estimate_T_closure(sigma_f)
    T = init_T
    i = 0  # global iteration counter
    # Running main loop
    print("\nRunning main loop...")
    while not optimizer.completed and i < max_iter:
        cur_hess_evals = optimizer.hess_evals
        rho_0 = rho / (T + 1)
        sigma_f = (1 / (2 * rho_0)) ** 0.5  # Budget for updating the estimate of T
        # ic(sigma_g)
        # input()
        optimizer.new_epoch_rho(rho_0)
        steps = min(T, update_T_every)
        flag = False
        cur_loss = 100
        for j in range(steps):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)
            loss = one_step(model, optimizer)
            if i % print_every == 0:
                print(f"Iteration {i:>5d}: {loss.item():>.5f}")

            if optimizer.completed:
                print("\nOptimization complete!")
                break
            if j % 10 == 0:
                # a heuristic to stop early (n is not large enough)
                last_loss = cur_loss
                cur_loss = loss.item()
                if cur_loss > last_loss:
                    print("Loss is increasing. Stopping...")
                    flag = True
                    break
        else:
            if T == min_T:
                print(
                    "With high probability, this should not happen since optimization should be completed by now!"
                )
                flag = True
            last_T = T
            T = estimate_T_closure(sigma_f)
            j += 1
            if T < min_T:
                T = min_T
            # print T change
            print(f"Update T: {last_T} -> {T}")
        rho -= (j + 1) * rho_0 - (j - (optimizer.hess_evals - cur_hess_evals)) * rho_0 / (
            2 + int(optimizer.line_search)) # (j - ...) could actually be j+1 but this minor difference doesn't matter (lower b)
        if flag:
            break

    return loss, rho, i


def opt_adapt_noise(
    model,
    optimizer,
    rho,
    init_T,
    min_T,
    max_iter,
    print_every,
    X=X,
    y=y,
    logging=True
):
    T = init_T
    i = 0
    last_epoch = False
    while not optimizer.completed and i < max_iter and T > 0:
        cur_hess_evals = optimizer.hess_evals
        rho_0 = rho / T
        sigma = optimizer.new_epoch_rho(rho_0)
        noise_g = 2**0.5 * 5 * sigma * opt_params["grad_sensitivity"]
        noise_H = 15 * sigma * opt_params["hess_sensitivity"]
        for j in range(T):
            i += 1
            # indices = rng.choice(n, size=batch_size)
            # XX, yy = X[indices], y[indices]
            # XX, yy = XX.to(device), yy.to(device)
            loss, (STEP, noisy_value) = one_step_extra(model, optimizer, X, y, logging=logging)
            if i % print_every == 0:
                print(f"Iteration {i:>5d}: {loss.item():>.5f}")

            if optimizer.completed:
                print("\nOptimization complete!")
                break

            if last_epoch or j > T / 2:
                continue

            if STEP == GRADIENT_STEP:
                if abs(noisy_value) < noise_g:
                    print(f"Gradient step: {abs(noisy_value):.5f} < {noise_g:.5f}")
                    break
            elif STEP == CURVATURE_STEP:
                if abs(noisy_value) < noise_H:
                    print(f"Curvature step: {abs(noisy_value):.5f} < {noise_H:.5f}")
                    break
        rho_prev = rho
        rho -= (j + 1) * rho_0 - (j + 1 - (optimizer.hess_evals - cur_hess_evals)) * rho_0 / (
            2 + int(optimizer.line_search))
        print(f"Privacy budget change: rho={rho_prev:.5f} -> {rho:.5f}")
        # if rho_prev < rho / 2:
        #     T = rho // rho_0 # run with remaining budget
        #     last_epoch = True
        # elif T / 2 > min_T:
        #     T //= 2
        #     print("Decrease noise, T:", T)
        # else:
        #     last_epoch = True
        if rho >= rho_prev / 2 and T / 2 > min_T:
            T //= 2
            print("Decrease noise, T:", T)
        else:
            T = int(rho // rho_0) # run with remaining budget
            last_epoch = True

    return loss, rho, i

def get_opt_method_name(strategy, line_search):
    method = "dpopt-" + strategy
    if line_search:
        method += "-ls"
    if batch_size:
        method += f"-batch={batch_size}"
    return method


def dpopt_exp(
    opt_params,
    strategy,
    initial_gap=None,
    rho=1,
    max_iter=1000,
    line_search=True,
    print_every=50,
):
    eps = rdp2dp(rho, 1 / n)
    method = get_opt_method_name(strategy, line_search)
    run_name = f"{method}-eps={eps:.1f}-eps_g={args.eps_g:.2f}-seed={args.seed}"
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=f"eps_g={args.eps_g},seed={args.seed}",
        config={
            "method": method,
            "seed": args.seed,
            "rho": rho,
            "eps": eps,
            "max_iter": max_iter,
            "line_search": line_search,
            "eps_g": args.eps_g,
            "eps_H": args.eps_g ** 0.5,
            "params": opt_params,
        },
        mode="online" if args.wandb else "disabled",
        reinit=True,
    )
    print(f">>>>>\nRunning {method} with rho={rho:.5f} (eps={eps:.2f}), seed={args.seed}\n")
    global rng
    rng = default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = ERM(opt_params["n_dim"], loss_fn, regularizer)
    optimizer = DPOPT(model.parameters(), opt_params, line_search=line_search)
    full_loss_closure = lambda: model(X, y).item()
    if initial_gap:
        gap = initial_gap
    else:
        rho_f = rho / max_iter  # TODO: just a simple heuristic for now
        rho -= rho_f
        sigma_f = (1 / (2 * rho_f)) ** 0.5
        gap = full_loss_closure() - opt_params["lower_bound"]
        gap += sigma_f * opt_params["f_sensitivity"] * abs(rng.normal(1))
    init_T = dp_estimate_T(opt_params, gap, line_search=line_search)
    print(f"gap={gap:.4f}, estimated T={init_T}")
    # min_T = int(init_T**0.5)  # TODO: another heuristic
    min_T = 20
    # measure running time
    start = time.perf_counter()
    if strategy == "adaptive":

        def estimate_T_closure(sigma_f):
            gap = full_loss_closure() - opt_params["lower_bound"]
            gap += sigma_f * opt_params["f_sensitivity"] * abs(rng.normal(1))
            return dp_estimate_T(opt_params, gap, line_search=line_search)

        out = opt_adapt_T(
            model,
            optimizer,
            rho,
            init_T,
            min_T,
            max_iter,
            print_every,
            estimate_T_closure,
            estimate_lb=True,
        )
    elif strategy == "adapt_noise":
        out = opt_adapt_noise(
            model, optimizer, rho, init_T, min_T, max_iter, print_every
        )
    elif strategy == "shrink":
        out = opt_shrink_T(
            model, optimizer, rho, init_T, min_T, max_iter, print_every, opt_params
        )
    elif strategy == "two_phase":
        out = opt_two_phase_T(
            model, optimizer, rho, init_T, max_iter, print_every
        )
    elif strategy == "fixed":
        out = opt_fixed_T(model, optimizer, rho, init_T, print_every)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    loss, rho_left, num_iter = out
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")

    if batch_size:
        # compute the loss on the full dataset
        optimizer.zero_grad()
        loss = model(X, y)
        loss.backward()
    grad_norm = model.w.grad.norm()
    # evaluate final hessian eigenvalue
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(X, y, w), model.w
    )
    hess_eig, _ = optimizer.smallest_eig(hess_closure())
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, hess_eig: {hess_eig:.5f} privacy budget left: {rho_left:.5f}"
    )
    # number of grad and hess evals
    print(
        f"Number of grad, hess evals: {optimizer.grad_evals}, {optimizer.hess_evals}"
    )
    # store loss, grad_norm, runtime in dictionary
    results = dict(
        loss=loss.item(),
        grad_norm=grad_norm.item(),
        hess_eig=hess_eig,
        runtime=runtime,
        rho_left=rho_left,
        num_iter=num_iter+1,
        grad_evals=optimizer.grad_evals,
        hess_evals=optimizer.hess_evals,
        completed=optimizer.completed,
    )
    run.summary.update(results)
    run.finish()
    return model, optimizer, results


def dptr_exp(
    opt_params,
    rho,
    initial_gap,
    print_every=50,
):
    eps = rdp2dp(rho, 1 / n)
    method = "dptr*"
    if batch_size:
        method += f"-batch={batch_size}"
    run_name = f"{method}-eps={eps:.1f}-eps_g={args.eps_g:.2f}-seed={args.seed}"
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=f"eps_g={args.eps_g},seed={args.seed}",
        config={
            "method": method,
            "seed": args.seed,
            "rho": rho,
            "eps": eps,
            "eps_g": args.eps_g,
            "eps_H": args.eps_g ** 0.5,
        },
        mode="online" if args.wandb else "disabled",
        reinit=True,
    )
    print(f">>>>>\nRunning {method} with rho={rho:.5f} (eps={eps:.2f}), seed={args.seed}\n")
    global rng
    rng = default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = ERM(n_dim, loss_fn, regularizer)

    # full_loss_closure = lambda: model(X, y).item()
    optimizer = DPTR(model.parameters(), opt_params, rho, initial_gap)
    T = optimizer.T
    wandb.config.max_iter = T
    print(opt_params)
    print(f"\nOptimize for at most T={T} iterations")
    start = time.perf_counter()
    for i in range(T):
        loss = one_step(model, optimizer)

        if optimizer.completed:
            print("\nOptimization complete!")
            break
        if i % print_every == 0:
            print(f"Iteration {i:>4d}: {loss.item():.5f}")
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")
    if batch_size:
        # compute the loss on the full dataset
        optimizer.zero_grad()
        loss = model(X, y)
        loss.backward()
    grad_norm = model.w.grad.norm()
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(X, y, w), model.w
    )
    hess_eig, _ = optimizer.smallest_eig(hess_closure())
    rho_left = rho * (1 - (i + 1) / T)
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, hess_eig: {hess_eig:.5f} privacy budget left: {rho_left:.5f}"
    )
    evals = i + 1
    print(
        f"Number of grad, hess evals: {evals}, {evals}"
    )
    results = dict(
        loss=loss.item(),
        grad_norm=grad_norm.item(),
        hess_eig=hess_eig,
        runtime=runtime,
        rho_left=rho_left,
        num_iter=evals,
        grad_evals=evals,
        hess_evals=evals,
        completed=optimizer.completed,
    )
    run.summary.update(results)
    run.finish()
    return model, optimizer, results

def dpgd_exp(
    opt_params,
    eps,
    initial_gap,
    print_every=50,
):
    method = "dpgd*"
    delta = 1 / n
    if batch_size:
        method += f"-batch={batch_size}"
    run_name = f"{method}-eps={eps:.1f}-eps_g={args.eps_g:.2f}-seed={args.seed}"
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=f"eps_g={args.eps_g},seed={args.seed}",
        config={
            "method": method,
            "seed": args.seed,
            "eps": eps,
            "delta": delta,
            "eps_g": args.eps_g,
            "eps_H": args.eps_g ** 0.5,
        },
        mode="online" if args.wandb else "disabled",
        reinit=True,
    )
    print(f">>>>>\nRunning {method} with alpha={args.eps_g:.5f} (eps={eps:.2f}), seed={args.seed}\n")
    global rng
    rng = default_rng(args.seed)
    torch.manual_seed(args.seed)
    
    model = ERM(n_dim, loss_fn, regularizer)
    # full_loss_closure = lambda: model(X, y).item()
    optimizer = DPGD(model.parameters(), opt_params, eps, delta, initial_gap)
    T = optimizer.T
    opt_params["T"] = T
    wandb.config.max_iter = T
    print(opt_params)
    print(f"\nOptimize for at most T={T} iterations")
    start = time.perf_counter()
    for i in range(T):
        loss = one_step(model, optimizer)

        if optimizer.completed:
            print("\nOptimization complete!")
            break
        if i % print_every == 0:
            print(f"Iteration {i:>5d}: {loss.item():.5f}")
    end = time.perf_counter()
    runtime = end - start
    print(f"Running time: {runtime:.2f}s")
    if batch_size:
        # compute the loss on the full dataset
        optimizer.zero_grad()
        loss = model(X, y)
        loss.backward()
    grad_norm = model.w.grad.norm()
    hess_closure = lambda: torch.autograd.functional.hessian(
        lambda w: model(X, y, w), model.w
    )
    hess_eig, _ = optimizer.smallest_eig(hess_closure())
    #TODO: compute rho_left
    rho_left = np.nan
    # rho_left = rho * (1 - (i + 1) / T)
    # print(
    #     f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, hess_eig: {hess_eig:.5f} privacy budget left: {rho_left:.5f}"
    # )
    print(
        f"Final loss: {loss:.5f}, grad_norm: {grad_norm:.5f}, hess_eig: {hess_eig:.5f}"
    )
    print(
        f"Number of grad, hess evals: {optimizer.grad_evals}, {optimizer.hess_evals}"
    )
    results = dict(
        loss=loss.item(),
        grad_norm=grad_norm.item(),
        hess_eig=hess_eig,
        runtime=runtime,
        rho_left=rho_left,
        num_iter=i+1,
        grad_evals=optimizer.grad_evals,
        hess_evals=optimizer.hess_evals,
        completed=optimizer.completed,
    )
    run.summary.update(results)
    run.finish()
    return model, optimizer, results 

# Note that the DPOPT algorithm outputs a ((1+c1)eps_g, (1+c)eps_H)-approximate second-order necessary point.

B_f, B_g, B_H, M = 1, 1, 1, 1
G = B_H

c2 = 1 / 10
c1 = c2 / M
c = 1 / 10
eps_g, eps_H = args.eps_g / (1 + c1), (M * args.eps_g)**0.5 / (1 + c)
# b_g, b_H = 10, 10
b_g, b_H = 20, 20

# fmt: off
opt_params = dict(
    alpha=args.eps_g, eps_g=eps_g, eps_H=eps_H, n_dim=n_dim, n=n, B_f=B_f, B_g=B_g, B_H=B_H, G=G, M=M, lower_bound=0.3, c=c, c1=c1, c2=1 / 12, b_g=b_g, beta_g=0.6, c_g=0.6, b_H=b_H, beta_H=0.5, c_H=0.25, checked=False,
    f_sensitivity = B_f / n,
    grad_sensitivity = 2 * B_g / n,
    hess_sensitivity = 2 * B_H * n_dim**0.5 / n,
)

check_and_compute_params(opt_params)
print_every = 10
initial_gap = 0.5

# rho2rdp_map = dict(zip(rhos, eps_lst))
# print(f"eps={eps:.2f}, rho={rho:.5f}")

def exp_range_eps(eps_lst, strategy_lst, rhos=None, run_dpopt=True, run_dptr=True, run_dpgd=False, delete_log=False):
    if rhos is None:
        rhos = dp2rdp(eps_lst, 1 / n)
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    with open(f"{args.dataset}-seed={args.seed}_eps_g={args.eps_g:.2f}_{time_str}.csv", 'w') as f:
        f.write("seed,method,eps,rho,completed,loss,grad_norm,hess_eig,runtime,rho_left,num_iter,grad_evals,hess_evals,\n")
        if run_dpopt:
            for eps, rho in zip(eps_lst, rhos):
                for strategy in strategy_lst:
                    # for ls in [True]:
                    for ls in [True, False]:
                        # try:
                        method = get_opt_method_name(strategy, ls)
                        model, optimizer, results = dpopt_exp(opt_params, strategy, initial_gap=initial_gap, rho=rho, max_iter=2000, line_search=ls, print_every=print_every)
                        print()
                        # write result to file
                        f.write(f"{args.seed},{method},{eps:.4f},{rho:.5f},{optimizer.completed},{results['loss']:.5f},{results['grad_norm']:.5f},{results['hess_eig']:.5f},{results['runtime']:.5f},{results['rho_left']:.5f},{results['num_iter']},{results['grad_evals']},{results['hess_evals']}\n")
                        # except Exception as e:
                        #     print(f"seed={args.seed}, method={method}, eps={eps:.4f}, rho={rho:.5f} failed:\n {str(e)}")
            # model, optimizer = dptr_exp(opt_params initial_gap=initial_gap, rho=rho, print_every=print_every)
        if delete_log:
            subprocess.run("rm -f wandb/*/logs/debug-internal.log", shell=True)
        if run_dptr:
            print("Running DPTR*...")
            for eps, rho in zip(eps_lst, rhos):
                method = "DPTR*" + (f"-batch={batch_size}" if batch_size else "")
                print(f">>>>>\nRunning {method} with rho={rho:.5f} (eps={eps:.2f})")
                model, optimizer, results = dptr_exp(opt_params, rho=rho, initial_gap=initial_gap, print_every=print_every)
                print()
                # write result to file
                f.write(f"{args.seed},{method},{eps:.4f},{rho:.5f},{optimizer.completed},{results['loss']:.5f},{results['grad_norm']:.5f},{results['hess_eig']:.5f},{results['runtime']:.5f}, {results['rho_left']:.5f},{results['num_iter']},{results['grad_evals']},{results['hess_evals']}\n")
        if run_dpgd:
            print("Running DPGD...")
            for eps, rho in zip(eps_lst, rhos):
                method = "DPGD" + (f"-batch={batch_size}" if batch_size else "")
                print(f">>>>>\nRunning {method} with rho={rho:.5f} (eps={eps:.2f})")
                model, optimizer, results = dpgd_exp(opt_params, eps, initial_gap=initial_gap, print_every=print_every)
                print()
                # write result to file
                f.write(f"{args.seed},{method},{eps:.4f},{rho:.5f},{optimizer.completed},{results['loss']:.5f},{results['grad_norm']:.5f},{results['hess_eig']:.5f},{results['runtime']:.5f}, {results['rho_left']:.5f},{results['num_iter']},{results['grad_evals']},{results['hess_evals']}\n")

        if delete_log:
            subprocess.run("rm -f wandb/*/logs/debug-internal.log", shell=True)


# eps_lst = np.arange(0.1, 0.5, 0.1)[::-1]

# eps_lst = np.array([0.2])
# eps_lst = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
eps_lst = np.arange(0.1, 1.1, 0.1)[::-1]
# eps_lst = np.array([0.7])
delta = 1 / n
rhos = dp2rdp(eps_lst, delta)

def exp(batch_sizes, run_dpopt=True, run_dptr=False, run_dpgd=False):
    global batch_size
    # strategies = ["two_phase", "shrink", "adapt_noise", "fixed"]
    # strategies = ["two_phase"]
    strategies = ["two_phase", "fixed"]
    # exp_range_eps(eps_lst, strategies, rhos, run_dptr=True)
    # seeds = [999]
        # for batch_size in [None]:
    for batch_size in batch_sizes:
        exp_range_eps(eps_lst, strategies, rhos, run_dpopt, run_dptr, run_dpgd)
        # subprocess.run(["wandb", "sync"], check=True)
    print(f"Done with seed {args.seed}\n\n")
        # wandb.alert(title="Runs finished", text=f"Runs finished for seed={SEED}")
        # for f in glob.glob("wandb/*/logs/debug-internal.log"):
        #     os.remove(f)

batch_sizes = [None, n // 100, n // 50]
# batch_sizes = [None]
# exp(batch_sizes, run_dpopt=False, run_dptr=False, run_dpgd=True)

if args.method == "all":
    exp(batch_sizes, run_dpopt=True, run_dptr=True)
elif args.method == "dpopt":
    exp(batch_sizes, run_dpopt=True, run_dptr=False)
elif args.method == "dptr":
    exp(batch_sizes, run_dpopt=False, run_dptr=True)

# exp(run_dpopt, run_dptr)
# exp(run_dpopt=False, run_dptr=True)
exit(0)

rho = rhos[0]
eps = eps_lst[0]
strategy = "adaptive"
strategy = "fixed"
strategy = "two_phase"

batch_size = 5000
batch_size = None
line_search = False
line_search = True
model, optimizer, results = dpopt_exp(opt_params, strategy, initial_gap=initial_gap, rho=rho, line_search=line_search, max_iter=2000,  print_every=print_every)
# model, optimizer, results = dptr_exp(opt_params, rho=rho, initial_gap=initial_gap, print_every=print_every)
# model, optimizer, results = dpgd_exp(opt_params, eps, initial_gap, print_every=print_every)
print(results)

# %%
