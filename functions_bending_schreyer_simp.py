# functions_bending_schreyer.py library file
# Benjamin Schreyer benontheplanet@gmail.com
#  stephan.schlamminger@nist.gov


import matplotlib.pyplot as plt
from mpmath import mpmathify
import mpmath as mp
import numpy as np
import math
import time
from scipy.interpolate import CubicSpline

# Implement placewise operations for mpmath matrices
def ov(op, *args):
    return mp.matrix(list(map(op, *args)))


# RK 45 iterator for mpmath library arbitrary precision
# ds_max is the fixed step size
# tol is not used currently
# f0: initial conditions, s0: initial time coordinate, s_final: last time coordinate, dfds: first order update function, tolerance unusued, ds_max: step size
##################
#"Speed and error"
##################
def mp_RKF45_fixed(f0, s0, s_final, dfds, step_tol, ds_max):
    # "time" coordinate, other variables, and their errors
    ss = []
    fs = []
    es = []

    # Initialize the iterated state
    f = mp.matrix((f0))
    s = s0

    ss.append(s)
    fs.append(f)
    es.append(f - f)  # Initial step error is 0

    # Runge-Kutta45 coefficient table for values and truncation errors
    Ak = mp.matrix(
        [
            mpmathify(0),
            mpmathify(2) / mpmathify(9),
            mpmathify(1) / mpmathify(3),
            mpmathify(3) / mpmathify(4),
            mpmathify(1),
            mpmathify(5) / mpmathify(6),
        ]
    )
    Ck = mp.matrix(
        [
            mpmathify(1) / mpmathify(9),
            mpmathify(0),
            mpmathify(9) / mpmathify(20),
            mpmathify(16) / mpmathify(45),
            mpmathify(1) / mpmathify(12),
            mpmathify(0),
        ]
    )
    CHk = mp.matrix(
        [
            mpmathify(47) / mpmathify(450),
            mpmathify(0),
            mpmathify(12) / mpmathify(25),
            mpmathify(32) / mpmathify(225),
            mpmathify(1) / mpmathify(30),
            mpmathify(6) / mpmathify(25),
        ]
    )
    CTk = mp.matrix(
        [
            mpmathify(1) / mpmathify(150),
            mpmathify(0),
            mpmathify(-3) / mpmathify(100),
            mpmathify(16) / mpmathify(75),
            mpmathify(1) / mpmathify(20),
            mpmathify(-6) / mpmathify(25),
        ]
    )
    Bkl = mp.matrix(
        [
            [
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(2) / mpmathify(9),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(1) / mpmathify(12),
                mpmathify(1) / mpmathify(4),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(69) / mpmathify(128),
                mpmathify(-243) / mpmathify(128),
                mpmathify(135) / mpmathify(64),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(-17) / mpmathify(12),
                mpmathify(27) / mpmathify(4),
                mpmathify(-27) / mpmathify(5),
                mpmathify(16) / mpmathify(15),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(65) / mpmathify(432),
                mpmathify(-5) / mpmathify(16),
                mpmathify(13) / mpmathify(16),
                mpmathify(4) / mpmathify(27),
                mpmathify(5) / mpmathify(144),
                mpmathify(0),
            ],
        ]
    )

    Ak = mp.matrix(
        [
            mpmathify(0),
            mpmathify(1) / mpmathify(4),
            mpmathify(3) / mpmathify(8),
            mpmathify(12) / mpmathify(13),
            mpmathify(1),
            mpmathify(1) / mpmathify(2),
        ]
    )
    Ck = mp.matrix(
        [
            mpmathify(25) / mpmathify(216),
            mpmathify(0),
            mpmathify(1408) / mpmathify(2565),
            mpmathify(2197) / mpmathify(4104),
            mpmathify(-1) / mpmathify(5),
            mpmathify(0),
        ]
    )
    CHk = mp.matrix(
        [
            mpmathify(16) / mpmathify(135),
            mpmathify(0),
            mpmathify(6656) / mpmathify(12825),
            mpmathify(28561) / mpmathify(56430),
            mpmathify(-9) / mpmathify(50),
            mpmathify(2) / mpmathify(55),
        ]
    )
    CTk = mp.matrix(
        [
            mpmathify(-1) / mpmathify(360),
            mpmathify(0),
            mpmathify(128) / mpmathify(4275),
            mpmathify(2197) / mpmathify(75240),
            mpmathify(-1) / mpmathify(50),
            mpmathify(-2) / mpmathify(55),
        ]
    )
    Bkl = mp.matrix(
        [
            [
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(1) / mpmathify(4),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(3) / mpmathify(32),
                mpmathify(9) / mpmathify(32),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(1932) / mpmathify(2197),
                mpmathify(-7200) / mpmathify(2197),
                mpmathify(7296) / mpmathify(2197),
                mpmathify(0),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(439) / mpmathify(216),
                mpmathify(8) / mpmathify(-1),
                mpmathify(3680) / mpmathify(513),
                mpmathify(-845) / mpmathify(4104),
                mpmathify(0),
                mpmathify(0),
            ],
            [
                mpmathify(-8) / mpmathify(27),
                mpmathify(2) / mpmathify(1),
                mpmathify(-3544) / mpmathify(2565),
                mpmathify(1859) / mpmathify(4104),
                mpmathify(-11) / mpmathify(40),
                mpmathify(0),
            ],
        ]
    )


    ks = [0, 0, 0, 0, 0, 0]
    # Fixed step iterations of Runge-Kutta saving the error and results
    while s < s_final - ds_max:
        delta = None

        args_s = s + ds_max * Ak

        truncation_error = None

        delta = None

        # Calculate the step change in the function
        # and the error estimate

        for k in range(6):
            fa = f
            for i in range(0, k):
                fa = fa + ks[i] * Bkl[k, i]
                #print(Bkl[k,i],end = ", ")
            #print("\n")
            ks[k] = ds_max * dfds(args_s[k], fa)

            if k == 0:
                delta = CHk[0] * ks[0]
                truncation_error = CTk[0] * ks[0]
            else:
                truncation_error = truncation_error + CTk[k] * ks[k]
                delta = delta + CHk[k] * ks[k]

        f = f + delta
        s +=  ds_max
        # Save the error
        es.append(truncation_error)
        ss.append(s)
        fs.append(f)

    return ss, fs, es


# Simple integration helper function for mpmath
def integrate_samples(grid, samples):
    return mp.fsum(samples) * (grid[1] - grid[0])

# variable spaced thetas along flexure convert to cartesian points by integrating
def integrate_xz(t, s):
    t = [float(x) for x in t]
    s = [float(x) for x in s]
    x = [0.0]
    z = [0.0]
    for i in range(len(s) - 1):

        x.append(np.sin(t[i]) * (s[i + 1] - s[i]) + x[-1])
        z.append(-np.cos(t[i]) * (s[i + 1] - s[i]) + z[-1])
    return np.array(x), np.array(z)



# Return S (the places along the fiber for which the bending angle is determined), F (list of vectors [M,theta,F2] where theta encodes the geometry of the bending, and Es is the errors
# Bending calculation for a zero moment zero theta initial flexure with "shape" Isamples
# grid: locations along the fiber the moment is sampled, Isamples: samples of the moment for bending along the fiber (same length as grid), order: not used, E: modulus of the material, Fsin: force coefficent for the sin term in bending, Fcos: boolean determining if a side force is present, theta0: intended bending angle, tol: tolerance for shooting
def bend_samples(
    grid, Isamples, order=4, E=1, Fsin=mpmathify(0), Fcos=mpmathify(0), M0 = 0, theta0=1, tol=0.001
):
    print("AUDIT THIS FUNCTION AGAIN BEFORE SERIOUS USE")
    # Useful for shorthand calculation since we dont have total numpy freedom with mpmath library
    onesmatrix = mp.matrix([1] * len(Isamples))

    #Spline interpolate the samples of the geometry to be compatible with RK intermediate sampling
    IS = CubicSpline(grid, Isamples, axis=0, bc_type="clamped", extrapolate=True)
    def I_spline(x):
        #print(IS(x))
        return IS(x)

    # Is the sine term present?
    Fs = not (Fsin == mpmathify(0))

    theta_is_small = mp.fabs(theta0) ** 2 / 2 < tol

    F1 = Fsin
    F2 = mpmathify(0)

    # Estimated scaling of the initial condition in the case of linear sin only bending
    ase = mpmathify(0)
    ###################################################
    # "Cosine term only" and "Cosine term only estimate"
    ###################################################
    # Case: cosine only bending "Cosine term only" and "Cosine term only estimate"


    

    # Case: sine only bending, "Sine term only", "Sine term only estimate"

    # Estimate the exponent using the linearized piecewise approximation for sine only bending
    integr = ov(mp.power, (Isamples), onesmatrix / mpmathify(-2))
    es = integrate_samples(grid, integr)

    ase = 1200#int(es * mp.sqrt(Fsin / E) * mp.log10(mp.exp(mpmathify(1))))
    F2 = Fcos
    #########################################


    # Now we have an estimate for initial parameters in any case (False: Need to implement mixed of cos and sin bending)
    # Do a run to see if we are close enough

    def dM_ds(t, Fco):
        return F1 * mp.sin(t) + Fco * mp.cos(t)

    def dt_ds(s, M):
        return M / E / I_spline(s)

    def df_ds(s, f):
        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0)])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    # Case: sine bending the estimate is not perfect, but should respond linearly so we can just rescale the estimate IC to scale the
    # final bending angle
    ##########################
    # "Sine term only estimate"
    ##########################
    if not Fcos:
        f0 = mp.matrix([mpmathify("1E-" + str(ase)), mpmathify(0), F2])
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify("1E-" + str(ase)) * theta0 / F[len(F) - 1][1], mpmathify(0), mpmathify(0)]
        )  # mpmathify()/F[len(F) - 1][1]
        if theta0 > 0.1:
            f0[0] = f0[0] /theta0 * 0.1
    else:
        f0 = mp.matrix([mpmathify(0), mpmathify(0),mpmathify("1E-" + str(ase))])
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print([M0, mpmathify(0), mpmathify("1E-" + str(ase)) * theta0 / F[len(F) - 1][1]])
        f0 = mp.matrix(
            [M0, mpmathify(0), mpmathify("1E-" + str(ase)) * theta0 / F[len(F) - 1][1]]
        )  # mpmathify()/F[len(F) - 1][1]
        if theta0 > 0.1:
            f0[2] = f0[2] /theta0 * 0.1
    print("SIMPLIFIED CODE, NO GUESSING")

    #############
    ###"Shooting"
    #############
    #shooting = regula_falsi_method_bending#secant_method_bending#regula_falsi_method_bending#secant_method_bending#binary_search_bending#
    # Sine only, which means shootign a small initial bending moment
    if not Fcos:
        #Define a function compatible with the default root finding. Use anderson as its fast and has convergence guarentee of regula falsi method
        def shot_function(x):
            ic = [0,0,0]
            ic[0] = x
            S, F, Es = mp_RKF45_fixed(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
            )
            return F[-1][1] - theta0
        print(f0[0], "IC, M(0)")
        f0[0] = mp.findroot(shot_function, (f0[0]/32, f0[0] * mpmathify(32)), solver="anderson", tol=tol*f0[0], verbose=True, verify=False)

        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print(f0, Fs, Fcos)
        return S, F, Es
    else:
        def shot_function(x):
            ic = [0,0,0]
            ic[2] = x
            ic[0] = M0
            S, F, Es = mp_RKF45_fixed(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
            )
            print(F[-1][1])
            return F[-1][1] - theta0
        print(f0[2], "IC, Fs")
        f0[2] = mp.findroot(shot_function, (f0[2]/32, f0[2] * mpmathify(32)), solver="anderson", tol=tol*f0[2], verbose=True, verify=False)

        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print(f0, Fs, Fcos)
        return S, F, Es
    # Cosine only, shoots the F2 term which is the force coefficient on the cosine force, also for the general case or both terms significant
    

