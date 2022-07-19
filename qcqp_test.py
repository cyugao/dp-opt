import cvxpy as cp
import numpy as np

""" Solve the QCQP
    min 0.5*x'Hx + g'x
    s.t. ||x||_2 <= r
"""

# Input data
n_dim = 50

# Form the problem
x = cp.Variable(n_dim)
H = np.random.randn(n_dim, n_dim)
H = H.T @ H
g = np.random.randn(n_dim)
I = np.eye(n_dim)
r2 = 1

# We now find the primal result and compare it to the dual result
# to check if strong duality holds i.e. the duality gap is effectively zero
constraint = 0.5 * cp.quad_form(x, I) <= 0.5 * r2
p = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, H) + g.T @ x),
    [constraint],
)
# [cp.atoms.pnorm(vec(x), 2) <= r])
primal_result = p.solve()

if p.status is cp.OPTIMAL:
    # Note that since our data is random, we may need to run this program multiple times to get a feasible primal
    # When feasible, we can print out the following values
    print(x.value)  # solution
    lam = constraint.dual_value
    dual_result = -0.5 * g.T @ np.linalg.solve(H + lam * I, g) - 0.5 * r2
    # ISSUE: dual result is matrix for some reason
    # print(dual_result)  # dual solution
    print("Our duality gap is:")
    print(primal_result - dual_result)
