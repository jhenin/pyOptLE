import numpy as np
import numba as nb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


def V_harm(x):
    k = 10.0
    return 0.5 * k * (x**2)

def nablaV_harm(x):
    k = 10.0
    return k * x

def V_dblwell(x):
    A = 0.05
    B = 1.2
    x2 = x**2
    return A * x2**2 - B * x2

def nablaV_dblwell(x):
    A = 0.05
    B = 1.2
    return 4. * A * x**3 - 2. * B * x

def overdamped_Langevin(n, N, dt, beta = 1, D = .1, nablaV = nablaV_harm):
    
    np.random.seed(1)

    # Start out of equilibrium
    x = np.zeros([n, N])

    x[:,0] = np.zeros(n) + np.random.normal(0, np.sqrt(2.*D*dt), size=n)
    for t in range(1,N):
        x[:,t] = x[:,t-1] - beta * D * nablaV(x[:,t-1]) * dt + np.random.normal(0, np.sqrt(2.*D*dt), size=n)
    return x


def load_traj(filename):
    # TODO handle case of separate traj files
    # Using colvars_traj module

    raw = np.loadtxt(filename)
    assert raw.shape[1] >= 2 and raw.shape[1] <= 3
    has_f = (raw.shape[1] == 3)

    t = 0
    old_t = 0
    q = list()
    f = list() if has_f else None
    qi = list()
    if has_f:
        fi = list()

    for r in raw:
        if r[0] < old_t:
            # Start new trajectory
            q.append(np.array(qi))
            qi = list()
            if has_f:
                f.append(np.array(fi))
                fi = list()
        qi.append(r[1])
        if has_f:
            fi.append(r[2])
        old_t = r[0]

    # Add last trajectory
    q.append(np.array(qi))
    if has_f:
        f.append(np.array(fi))

    return q, f


def history(res):
    history.opt.append(res)
    if history.nsteps%100 == 0:
        print('Step', history.nsteps, np.square(res).sum())
    history.nsteps += 1


def optimize_model(initial_params, knots, q, deltaq, f, dt=1, T=300, RT=None):
    # kcal/mol
    R = 0.00198720425864
    if RT is not None:
        RT = 1.
    else:
        RT = R * T
    beta = 1 / RT

    history.opt = list()
    history.nsteps = 0

    start_time = time.time()
    # Minimize the objective function using L-BFGS-B algorithm
    result = minimize(objective_order1_debiased, initial_params,  args=(knots, q, deltaq, dt, beta, f), jac=True, method='L-BFGS-B', callback=history)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(result.nit, result.message)


# WARNING - assumes fixed-size arrays - need to re-run cell each time data sizes change to recompile JIT
# What happens once this is in a module?

# TODO do search only once and pass indexes, h values

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
def objective_order1_debiased(params, knots, q, deltaq, dt, beta, f):
    """Objective function: order-1 OptLE for overdamped Langevin, order-1 propagator
    Includes the debiasing feature of Hallegot, Pietrucci and Hénin for time-dependent biases

    Args:
        params (ndarray): parameters of the model - piecewise-linear grad F (free energy) and log D
        knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF
        q (list of ndarray): trajectories of the CV
        deltaq (list of ndarray): trajectories of CV differences
        f (list of ndarray): trajectories of the biasing force

    Returns:
        real, ndarray: objective function and its derivatives with respect to model parameters
    """

    dim = params.shape[0]//2
    # Accumulators
    logL = 0.0
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