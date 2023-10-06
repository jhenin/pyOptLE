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

    x[:,0] = np.zeros(n)

    for t in range(1,N):
        x[:,t] = x[:,t-1] - beta * D * nablaV(x[:,t-1]) * dt + np.random.normal(0, np.sqrt(2.*D*dt), size=n)
    return x

def load_traj(filenames):
    # TODO handle case of separate traj files
    # Using plot_colvar_traj module

    if type(filenames) == str:
        filenames = [filenames]

    q = list()
    f = list()

    for filename in filenames:
        raw = np.loadtxt(filename)
        assert raw.shape[1] >= 2 and raw.shape[1] <= 3
        has_f = (raw.shape[1] == 3)

        t = 0
        old_t = 0

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
    if not has_f:
        f = None
    return q, f


def history(res):
    history.opt.append(res)
    if history.nsteps%100 == 0:
        print('Step %10i    Parameter mean sq. %10.2f' % (history.nsteps, np.mean(np.square(res))))
    history.nsteps += 1


def optimize_model(initial_params, knots, q, f=None, dt=1., T=300., RT=None, use_midpoint=False):
    '''Optimize an OptLE model given a trajectory.
    This assumes Overdamped Langevin dynamics.
    It uses an order-1 development of the short-term propagator.

    Args:
        initial_params (ndarray): initial parameters of the model
        knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF.
        q (list of ndarray): trajectories of the CV.
        f (list of ndarray): trajectories of the biasing force.
        dt (float, optional): time increment. Defaults to 1.
        T (float, optional): Temperature (K). Defaults to 300.
        RT (float, optional): RT. Defaults to None.

    Returns:
        _type_: _description_
    '''

    obj = objective_order1_debiased

    R = 0.00198720425864 # kcal/mol
    if RT is not None:
        RT = 1.
    else:
        RT = R * T
    beta = 1 / RT

    print("Pre-processing trajectories...")
    start_time = time.time()
    traj = preprocess_traj(q, knots, use_midpoint)
    print("Done in %.3f seconds" % (time.time() - start_time))

    if f is None:
        f = [np.zeros_like(qi) for qi in q]
    f = nb.typed.List(f)

    print("Compiling and running objective function...")
    start_time = time.time()
    print('Obj = %.3f' % obj(initial_params, knots, traj, dt, beta, f)[0])
    print("Compiled and ran in %.3f seconds" % (time.time() - start_time))

    start_time = time.time()
    print('Obj = %.3f' % obj(initial_params, knots, traj, dt, beta, f)[0])
    print("Runs in %.3f seconds" % (time.time() - start_time))

    # Record optimizer history
    history.opt = list()
    history.nsteps = 0

    start_time = time.time()
    # Minimize the objective function using L-BFGS-B algorithm
    result = minimize(obj, initial_params,  args=(knots, traj, dt, beta, f), jac=True, method='L-BFGS-B', callback=history)

    print(result.message)
    print("Optimization converged in %i steps, %.3f seconds" % (result.nit, time.time() - start_time))

    return result, history.opt


def preprocess_traj(q, knots, use_midpoint = False):
    '''Preprocess colvar trajectory with a given grid for faster model optimization

    Args:
        q (list of ndarray): trajectories of the CV.
        knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF.

    Returns:
        traj (numba types list): list of tuples (bin indices, bin positions, displacements)
    '''

    # TODO: enable subsampling by *averaging* biasing force in interval
    # Then run inputting higher-res trajectories

    traj = list()

    for qi in q:
        deltaq = qi[1:] - qi[:-1]

        if use_midpoint:
            # Use mid point of each interval
            # Implies a "leapfrog-style" integrator that is not really used for overdamped LE
            ref_q = 0.5 * (qi[:-1] + qi[1:])
        else:
            # Truncate last traj point to match deltaq array
            ref_q = qi[:-1]

        # bin index on possibly irregular grid
        idx = np.searchsorted(knots,ref_q)

        assert (idx > 0).all() and (idx < len(knots)).all(), 'Out-of-bounds point(s) in trajectory\n'
        # # Other option: fold back out-of-bounds points - introduces biases
        # idx = np.where(idx == 0, 1, idx)
        # idx = np.where(idx == len(knots), len(knots) - 1, idx)

        q0, q1 = knots[idx - 1], knots[idx]
        # fractional position within the bin
        h = (qi[:-1] - q0) / (q1 - q0)

        traj.append((idx, h, deltaq))

    # Numba prefers typed lists
    return nb.typed.List(traj)



# WARNING - assumes fixed-size arrays - need to re-run cell each time data sizes change to recompile JIT
# What happens once this is in a module?

@nb.njit
def linear_interpolation_with_gradient(idx, h, knots, fp):

    n_knots = knots.shape[0]
    x0, x1 = knots[idx - 1], knots[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    # Second parameter set is in second half of array
    g0, g1 = fp[idx - 1 + n_knots], fp[idx + n_knots]

    hm = 1 - h
    val_f = hm * f0 + h * f1
    val_g = hm * g0 + h * g1

    # Set gradient elements one by one
    grad = np.zeros((knots.shape[0], idx.shape[0]))
    for i, ik in enumerate(idx):
        grad[ik - 1, i] = hm[i]
        grad[ik, i] = h[i]
    return val_f, val_g, grad


@nb.njit(parallel=True)
def objective_order1_debiased(params, knots, traj, dt, beta, f):
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

    # The loop over trajectories is parallelized by Numba
    for i in nb.prange(len(traj)):
        idx, h, deltaq = traj[i]
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, knots, params)
        # dXdk is the gradient with respect to the knots (same for all quantities)

        # Debiasing (truncate last traj point)
        G -= f[i][:-1]
    
        phi = - beta * np.exp(logD) * G * dt
        dphidlD = - beta * np.exp(logD) * G * dt
        dphidG =  - beta * np.exp(logD) * dt
        
        mu = 2.0 * np.exp(logD) * dt
        dmudlD = 2.0 * np.exp(logD) * dt
        logL += (0.5 * logD + np.square(deltaq - phi) / (2.0 * mu)).sum()
        dLdlD = 0.5 + (2.0 * (deltaq - phi) * -1.0 * dphidlD * (2.0 * mu) - np.square(deltaq - phi) * 2.0 * dmudlD) / np.square(2.0 * mu)
        dLdG = 2.0 * (deltaq - phi) * -1.0 * dphidG / (2.0 * mu)

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



def check_gradient(obj, q, f, delta = 1e-6):
    n_knots = 10
    # Create grid and initial parameter set
    epsilon = 1e-10
    qmin_data = min([np.min(qi) for qi in q]) - epsilon
    qmax_data = max([np.max(qi) for qi in q]) + epsilon
    qmin = qmin_data
    qmax = qmax_data
    knots = np.linspace(qmin, qmax, n_knots)

    # Initial guess for the parameters: zero free energy gradient, constant D = 1
    initial_params = np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.))
    traj = preprocess_traj(q, knots)

    grad, g = finite_diff(initial_params, delta, objective_order1_debiased, knots, traj, 1, 1, f)

    rms = np.sqrt(np.mean(np.square(grad)))
    rmsd = np.sqrt(np.mean(np.square(grad - g)))

    print('RMS gradient: %f     RMS difference: %f' % (rms, rmsd))


def finite_diff(x, dx, objective, *args):
    
    grad = np.zeros_like(x)
    y1, g = objective(x, *args)

    for i in range(x.shape[0]):
        x2 = np.copy(x)
        x2[i] += dx
        y2, _ = objective(x2, *args)
        grad[i] = (y2-y1) / dx
    return grad, g