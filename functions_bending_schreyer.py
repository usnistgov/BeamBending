# functions_bending_schreyer.py library file
# Benjamin Schreyer benontheplanet@gmail.com

#TODO:
#RungeKutta aribtrary precision in compiled language
#Show numerical ansats works with the solutions I make
#Paper plots, simple example (uniform), save the plot data as txt not just code
#Illinois method to speed up some edge cases and still guarentee convergence?

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
    ss = [-1] * (int(s_final / ds_max) + 1)
    fs = [-1] * (int(s_final / ds_max) + 1)
    es = [-1] * (int(s_final / ds_max) + 1)

    # Initialize the iterated state
    f = mp.matrix((f0))
    s = s0

    ss[0] = s
    fs[0] = f
    es[0] = f - f  # Initial step error is 0

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


    ks = [0, 0, 0, 0, 0, 0]
    # Fixed step iterations of Runge-Kutta saving the error and results
    for w in range(int(s_final / ds_max)):
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
        es[w + 1] = truncation_error
        ss[w + 1] = s
        fs[w + 1] = f

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

# Binary search shooting of a bending problem, using the cosine force term or the initial small M depending on the type of bending
# L: left limit of search space, R: right limit of search space, grid: sample locations along the fiber, df_ds: evolution function for bending, tol:tolerance, theta0: goal final angle to be found by shooting, search_param: parameter to modify in attempting to shoot, for no cosine term this is M(0), for a cosine term the cosine coefficient is the symmetry breaking initial condition causing a bend away from straight.
def binary_search_bending(L, R, grid, df_ds, tol, theta0, search_param):
    s0 = grid[0]

    # initial condition on the shooting parameter is at the center
    # of the binary search interval
    ic = [0, 0, 0]
    ic[search_param] = (L + R) / mpmathify(2)

    # Trial the center interval initial condition
    S, F, Es = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )

    # If tolerance is met in the shooting parameter or interval size end
    # calculation
    print(L,R, F[-1][1])
    if mp.fabs(theta0 - F[-1][1]) / theta0 < tol or mp.fabs(L - R) < tol * mp.fabs(
        L + R
    ):
        return (L + R) / mpmathify(2)

    # Left and right interval binary search branches
    if F[-1][1] > theta0:
        return binary_search_bending(
            L, (L + R) / mpmathify(2), grid, df_ds, tol, theta0, search_param
        )

    if F[-1][1] < theta0:
        return binary_search_bending(
            (L + R) / mpmathify(2), R, grid, df_ds, tol, theta0, search_param
        )  # return binary_search_bending((L+R)/mpmathify(2), R,EXP,I_z,key, tol)

    # catch all/error case
    return (L + R) / mpmathify(2)
def regula_falsi_method_bending(L, R, grid, df_ds, tol, theta0, search_param):
    s0 = grid[0]
    ic = [0, 0, 0]

    # Initial guesses for Regula Falsi method
    ic[search_param] = L
    S1, F1, Es1 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    ic[search_param] = R
    S2, F2, Es2 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )

    # First value at R
    F1_value = F1[-1][1]
    F2_value = F2[-1][1]
    #if F1_value < mpmathify(0):
        #F1_value = mp.fabs(F1_value)
    #if F2_value < mpmathify(0):
        #F2_value = mp.fabs(F2_value)

    # Continue iterating using the Regula Falsi method
    iteration = 0
    while mp.fabs(F1_value - theta0) / theta0 >= tol and mp.fabs(L - R) >= tol * (
        L + R
    ):

        # Check for division by zero
        if F2_value - F1_value == 0:  # Prevent division by zero
            return (L + R) / mp.mpf(2)  # Return midpoint if derivative is zero

        # Calculate the new guess using the Regula Falsi formula
        new_L = R - (F2_value - theta0) * (R - L) / (F2_value - F1_value)

        # Update values based on the new guess
        ic[search_param] = new_L
        S_new, F_new, Es_new = mp_RKF45_fixed(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        
        F_new_value = F_new[-1][1]
        #if F_new_value < mpmathify(0):
            #F_new_value = mp.fabs(F_new_value)
        print(L, R, F_new_value,"regula falsi")
        
        # Update variables for the next iteration
        # The key difference here from secant method is that we check if F_new_value is closer to zero
        if F_new_value * F1_value < 0:
            R = new_L
            F2_value = F_new_value
        else:
            L = new_L
            F1_value = F_new_value

        iteration += 1

    return L

def illinois_method_bending(L, R, grid, df_ds, tol, theta0, search_param):
    s0 = grid[0]
    ic = [0, 0, 0]

    # Initial guesses for Illinois method
    ic[search_param] = L
    S1, F1, Es1 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    ic[search_param] = R
    S2, F2, Es2 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )

    # First values at L and R
    F1_value = F1[-1][1]
    F2_value = F2[-1][1]

    # Continue iterating using the Illinois method
    iteration = 0
    while mp.fabs(F1_value - theta0) / theta0 >= tol and mp.fabs(L - R) >= tol * (L + R):

        # Check for division by zero
        if F2_value - F1_value == 0:  # Prevent division by zero
            return (L + R) / mp.mpf(2)  # Return midpoint if derivative is zero

        # Illinois method modification: interpolate to find the new guess
        new_L = R - (F2_value - theta0) * (R - L) / (F2_value - F1_value)

        # Update values based on the new guess
        ic[search_param] = new_L
        S_new, F_new, Es_new = mp_RKF45_fixed(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )

        F_new_value = F_new[-1][1]
        print(L, R, F_new_value, "illinois method")

        # Illinois method logic: adjust bounds to accelerate convergence
        if F_new_value * F1_value < 0:  # Root lies between new_L and R
            R = new_L
            F2_value = F_new_value
        elif F_new_value * F2_value < 0:  # Root lies between L and new_L
            L = new_L
            F1_value = F_new_value
        else:  # Same sign, move one bound to the middle point
            if F1_value * F2_value < 0:  # One of the signs is near zero, adjust L or R
                if abs(F1_value) < abs(F2_value):
                    R = new_L  # Adjust R if F1_value is closer to zero
                    F2_value = F_new_value
                else:
                    L = new_L  # Adjust L if F2_value is closer to zero
                    F1_value = F_new_value

        iteration += 1

    return L

def secant_method_bending(L, R, grid, df_ds, tol, theta0, search_param):
    s0 = grid[0]
    ic = [0, 0, 0]

    # Initial guesses for secant method
    ic[search_param] = L
    S1, F1, Es1 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    ic[search_param] = R
    S2, F2, Es2 = mp_RKF45_fixed(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )

    # First value at R
    F1_value = F1[-1][1]
    F2_value = F2[-1][1]
    #if F1_value < mpmathify(0):
        #F1_value = mp.fabs(F1_value)
    #if F2_value < mpmathify(0):
        #F2_value = mp.fabs(F2_value)
                
    # Continue iterating using the secant method
    iteration = 0
    while mp.fabs(F2_value - theta0) / theta0 >= tol and mp.fabs(L - R) >= tol * (
        L + R
    ):

        # Update guess using the secant method formula
        if F2_value - F1_value == 0:  # Prevent division by zero
            return (L + R) / mp.mpf(2)  # Return midpoint if derivative is zero
        new_L = R - (F2_value - theta0) * (R - L) / (F2_value - F1_value)

        # Update values based on the new guess
        ic[search_param] = new_L
        S_new, F_new, Es_new = mp_RKF45_fixed(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        
        F_new_value = F_new[-1][1]
        #if F_new_value < mpmathify(0):
            #F_new_value = mp.fabs(F_new_value)
        print(L,R, F_new_value)
        # Update variables for the next iteration
        #if(new_L < mpmathify(0)):
            #new_L = mpmathify(0)
        L = R
        F1_value = F2_value
        R = new_L
        F2_value = F_new_value

        iteration += 1


    return R


# Return S (the places along the fiber for which the bending angle is determined), F (list of vectors [M,theta,F2] where theta encodes the geometry of the bending, and Es is the errors
# Bending calculation for a zero moment zero theta initial flexure with "shape" Isamples
# grid: locations along the fiber the moment is sampled, Isamples: samples of the moment for bending along the fiber (same length as grid), order: not used, E: modulus of the material, Fsin: force coefficent for the sin term in bending, Fcos: boolean determining if a side force is present, theta0: intended bending angle, tol: tolerance for shooting
def bend_samples(
    grid, Isamples, order=4, E=1, Fsin=mpmathify(0), Fcos=True, theta0=1, tol=0.001
):
    # Useful for shorthand calculation since we dont have total numpy freedom with mpmath library
    onesmatrix = mp.matrix([1] * len(Isamples))

    #Spline interpolate the samples of the geometry to be compatible with RK intermediate sampling
    I_spline = CubicSpline(grid, Isamples, axis=0, bc_type="clamped", extrapolate=True)

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

    if Fcos and (not Fs):
        # Analytic linear guess for cosine only bending
        I_cos = integrate_samples(grid, ov(mp.fdiv, (grid / E), Isamples))

        # Take a small angle guess before doing large angle shooting
        F2 = theta0 / I_cos

    # Case: sine only bending, "Sine term only", "Sine term only estimate"
    if Fs and (not Fcos):
        # Estimate the exponent using the linearized piecewise approximation for sine only bending
        integr = ov(mp.power, (Isamples), onesmatrix / mpmathify(-2))
        es = integrate_samples(grid, integr)

        ase = int(es * mp.sqrt(Fsin / E) * mp.log10(mp.exp(mpmathify(1))))
    #########################################
    # "General case", "Both terms significant"
    #########################################
    # Mixed estimate "General case", "Both terms significant"
    if Fcos and Fs:
        # Analytic linear guess for cosine only bending
        I_cos = integrate_samples(grid, ov(mp.fdiv, (grid / E), Isamples))

        # Take a small angle guess before doing large angle shooting
        F2 = theta0 / I_cos

        # Estimate the exponent using the linearized piecewise approximation for sine only bending
        integr = ov(mp.power, (Isamples), onesmatrix / mpmathify(-2))
        es = integrate_samples(grid, integr)

        ase = int(es * mp.sqrt(Fsin / E) * mp.log10(mp.exp(mpmathify(1))))

        F2 = mpmathify("1E-" + str(ase)) * F2

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
    if Fcos:  # and (not Fs):
        f0 = mp.matrix([0, 0, F2])
    s0 = grid[0]

    # Case: sine bending the estimate is not perfect, but should respond linearly so we can just rescale the estimate IC to scale the
    # final bending angle
    ##########################
    # "Sine term only estimate"
    ##########################
    if Fs and (not Fcos):
        f0 = mp.matrix([mpmathify("1E-" + str(ase)), mpmathify(0), F2])
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify("1E-" + str(ase)) * theta0 / F[len(F) - 1][1], mpmathify(0), F2]
        )  # mpmathify()/F[len(F) - 1][1]
        
    #########################################
    # "General case", "Both terms significant"
    #########################################
    if Fs and Fcos:
        f0 = mp.matrix([mpmathify(0), mpmathify(0), F2])
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify(0), mpmathify(0),F2 * theta0 / F[len(F) - 1][1]]
        )  
    
    # Large angle we now need to binary search to find the initial condition by shooting
    #############
    ###"Shooting"
    #############
    shooting = regula_falsi_method_bending#secant_method_bending#regula_falsi_method_bending#secant_method_bending#binary_search_bending#
    # Sine only, which means shootign a small initial bending moment
    if Fs and (not Fcos):
        f0[0] = shooting(
            mpmathify(0), f0[0] * mpmathify(32), grid, df_ds, tol, theta0, 0
        ) 
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        return S, F, Es
    # Cosine only, shoots the F2 term which is the force coefficient on the cosine force, also for the general case or both terms significant
    if Fcos:  # (not Fs) and
        f0[2] = shooting(
            mpmathify(0), f0[2] * mpmathify(32), grid, df_ds, tol, theta0, 2
        )  
        S, F, Es = mp_RKF45_fixed(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        # Return S (the places along the fiber for which the bending angle is determined), F (list of vectors [M,theta,F2] where theta encodes the geometry of the bending, and Es is the errors
        return S, F, Es


