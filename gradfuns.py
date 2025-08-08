import mpmath as mp
from random import choice

def _auto_h(xi, dps=None):
    # Step for finite differences: ~ eps^(1/3) * max(1, |xi|)
    # (good balance between cancellation and truncation)
    if dps is None:
        eps = mp.eps
    else:
        # crude: eps ~ 10^(-dps)
        eps = mp.mpf(10) ** (-dps)
    return (eps ** (mp.mpf(1)/3)) * max(1, abs(xi))

def _grad_central(f, x, dps=None):
    n = len(x)
    g = [mp.mpf(0)] * n
    fx_cache = None
    for i in range(n):
        h = _auto_h(x[i], dps)
        ei = [mp.mpf(0)] * n
        ei[i] = h
        xp = [x[j] + ei[j] for j in range(n)]
        xm = [x[j] - ei[j] for j in range(n)]
        fp = f(xp)
        fm = f(xm)
        g[i] = (fp - fm) / (2*h)
    return g

def _grad_spsa(f, x, dps=None):
    # Simultaneous Perturbation Stochastic Approximation
    n = len(x)
    # perturbation magnitude ~ eps^(1/3) * scale
    scale = max(1, max(abs(xi) for xi in x))
    if dps is None:
        eps = mp.eps
    else:
        eps = mp.mpf(10) ** (-dps)
    c = (eps ** (mp.mpf(1)/3)) * scale

    # Bernoulli ±1 directions
    delta = [mp.mpf(choice([-1, 1])) for _ in range(n)]

    xp = [x[i] + c*delta[i] for i in range(n)]
    xm = [x[i] - c*delta[i] for i in range(n)]
    fp = f(xp)
    fm = f(xm)

    g = [(fp - fm) / (2*c*delta[i]) for i in range(n)]
    return g

def gd(
    f,
    x0,
    step0=mp.mpf('1.0'),
    max_iter=200,
    tol=mp.mpf('1e-8'),
    grad='central',        # 'central' or 'spsa'
    line_beta=mp.mpf('0.5'),   # backtracking factor
    line_sigma=mp.mpf('1e-4'), # Armijo condition constant
    dps=None,
    verbose=False
):
    """
    Derivative-free gradient descent using only function evaluations.

    Args:
      f: function R^n -> R, takes a list/tuple of mp.mpf and returns mp.mpf
      x0: initial point (list/tuple of numbers)
      step0: initial step size for line search upper bound
      grad: 'central' (2n evals) or 'spsa' (2 evals)
      tol: stop when ||grad||_inf <= tol or step becomes tiny
      dps: set mp.mp.dps for internal precision (optional)
    Returns:
      x, fx, info
    """
    if dps is not None:
        mp.mp.dps = int(dps)

    x = [mp.mpf(v) for v in x0]
    fx = f(x)

    grad_fn = _grad_central if grad == 'central' else _grad_spsa

    for it in range(1, max_iter+1):
        g = grad_fn(f, x, dps)
        gnorm_inf = max(abs(gi) for gi in g)
        if verbose:
            print(f"it={it:4d}  f={fx!s}  ||g||_inf={gnorm_inf!s}")

        if gnorm_inf <= tol:
            return x, fx, {'iter': it, 'grad_norm_inf': gnorm_inf, 'converged': True}

        # Descent direction
        d = [-gi for gi in g]

        # Armijo backtracking
        t = mp.mpf(step0)
        # Optional simple upper bound to avoid huge steps
        # t = min(t, mp.mpf('1.0')/max(mp.mpf('1e-30'), gnorm_inf))
        # Armijo: f(x + t d) <= f(x) + sigma t g·d
        gd_dot = sum(g[i]*d[i] for i in range(len(x)))  # = -||g||^2 <= 0
        # If gradient is (numerically) zero, stop
        if abs(gd_dot) < mp.mpf('1e-50'):
            return x, fx, {'iter': it, 'grad_norm_inf': gnorm_inf, 'converged': True}

        while True:
            x_new = [x[i] + t*d[i] for i in range(len(x))]
            fx_new = f(x_new)
            if fx_new <= fx + line_sigma * t * gd_dot:
                # sufficient decrease
                x, fx = x_new, fx_new
                break
            t *= line_beta
            if t < mp.mpf('1e-40'):
                # step collapsed; treat as convergence stall
                return x, fx, {'iter': it, 'grad_norm_inf': gnorm_inf, 'converged': False, 'reason': 'line search stalled'}

    return x, fx, {'iter': max_iter, 'grad_norm_inf': gnorm_inf, 'converged': False, 'reason': 'max_iter'}
