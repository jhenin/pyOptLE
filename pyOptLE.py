import numpy as np
import numba as nb
from scipy.optimize import minimize


# TODO pass dt
dt = 0.1
RT = 1.
beta = 1/RT


# WARNING - assumes fixed-size arrays
# Need to re-run cell each time data sizes change to recompile JIT

@nb.njit(parallel=False)
def linear_interpolation_with_gradient(x, xp, fp):
    idx = np.searchsorted(xp, x)
    # Assuming we are within bounds, or we get errors
    # Faster without checks...
    # idx = np.where(idx == 0, 1, idx)  # If idx == 0, set it to 1
    # idx = np.where(idx == len(xp), len(xp) - 1, idx)  # If idx == len(xp), set it to len(xp) - 1
    # assert (idx > 0).all() and (idx < len(xp)).all(), 'linear_interpolation_with_gradient: out of bounds \n' + str(x) + '\n' + str(xp)
    n_knots = xp.shape[0]
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    # Second parameter set in second half of array
    g0, g1 = fp[idx - 1 + n_knots], fp[idx + n_knots]

    hm = (x1 - x) / (x1 - x0)
    hp = 1 - hm
    val_f = hm * f0 + hp * f1
    val_g = hm * g0 + hp * g1

    # Set gradient elements one by one
    grad = np.zeros((xp.shape[0], x.shape[0]))
    for i, ik in enumerate(idx):
        grad[ik - 1, i] = hm[i]
        grad[ik, i] = hp[i]        
    return val_f, val_g, grad


@nb.njit(parallel=True)
def objective(params, q, deltaq, f, knots):

    # Accumulators
    logL = 0.0
    dim = params.shape[0]//2
    dlogLdklD = np.zeros(dim)
    dlogLdkG = np.zeros(dim)

    for i in nb.prange(len(q)):
        G, logD, dXdk = linear_interpolation_with_gradient(q[i][:-1], knots, params)
        # dXdk is the gradient with respect to the knots (same for all quantities)
        
        # Debiasing
        G -= f[i][:-1]
    
        phi = - beta * np.exp(logD) * G * dt
        dphidlD = - beta * np.exp(logD) * G * dt
        dphidG =  - beta * np.exp(logD) * dt
        
        mu = 2.0 * np.exp(logD) * dt
        dmudlD = 2.0 * np.exp(logD) * dt

        logL += (0.5 * logD + np.square(deltaq[i] - phi) / (2.0 * mu)).sum()
        dLdlD = 0.5 + (2.0 * (deltaq[i] - phi) * -1.0 * dphidlD * (2.0 * mu) - np.square(deltaq[i] - phi) * 2.0 * dmudlD) / np.square(2.0 * mu)
        dLdG = 2.0 * (deltaq[i] - phi) * -1.0 * dphidG / (2.0 * mu)

        dlogLdkG += np.dot(dXdk, dLdG)
        dlogLdklD += np.dot(dXdk, dLdlD)

    return logL, np.concatenate((dlogLdkG, dlogLdklD))


def piecewise_linear_int(x, knots, optimized_G):
    # Do not compute integral for first value, it is arbitrary
    sub_xs = np.split(x[1:], (len(knots) - 1))
    # Arbitrary boundary condition for integration
    start = 0.
    pred = [start]

    for i in range(len(knots)-1):
        x_i = knots[i]
        x_i1 = knots[i+1]
        dgrad_dx = (optimized_G[i+1] - optimized_G[i]) / (x_i1 - x_i)
        grad_i = optimized_G[i]
        sub_x = sub_xs[i]
        sub_pred = 0.5 * dgrad_dx * np.square(sub_x) + (grad_i - dgrad_dx * x_i) * sub_x
        # Extrapolate to previous knot for boundary condition
        my_start = (grad_i - 0.5 * dgrad_dx * x_i) * x_i
        shift = start - my_start
        sub_pred += shift
        # Remember initial condition for next window
        start = sub_pred[-1]
        pred += list(sub_pred)

    # Real BC: min = 0
    return pred - np.min(np.array(pred))


def finite_diff(x, dx, objective, **args):
    
    grad = np.zeros_like(x)
    y1, g = objective(x, **args)

    for i in range(x.shape[0]):
        x2 = np.copy(x)
        x2[i] += dx
        y2, _ = objective(x2, **args)
        grad[i] = (y2-y1) / dx
    return grad, g