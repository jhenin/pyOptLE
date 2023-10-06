import numpy as np
import time
from scipy.optimize import minimize
import pyOptLE as optle

T = 300
dt = 0.1
n_knots = 20

filename = 'data_dblwl/dblwl_abmd/biased/dblwl_h7_k05_g10_abmd'

q, f = optle.load_traj(filename)

epsilon = 1e-10
qmin = min([np.min(qi) for qi in q]) - epsilon
qmax = max([np.max(qi) for qi in q]) + epsilon
knots = np.linspace(qmin, qmax, n_knots)

initial_params = np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.))


# Optimize with debiasing

result, hist = optle.optimize_model(initial_params, knots, q, f, dt, T = T)

print(result.nit, result.message)
print(result.x)

