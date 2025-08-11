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
from rungekuttacoefficients import *

# Implement placewise operations for mpmath matrices
def ov(op, *args):
    return mp.matrix(list(map(op, *args)))


# RK 45 iterator for mpmath library arbitrary precision
# ds_max is the fixed step size
# tol is not used currently
# f0: initial conditions, s0: initial time coordinate, s_final: last time coordinate, dfds: first derivative vector returning function, tolerance unusued, ds_max: step size
##################
#"Speed and error"
##################

#A version of the RK45_fixed that uses a variable step size to reach the desired tolerance.
def mp_RKF45_adaptive(f0, s0, s_final, dfds, step_tol, ds_max):
    #I
    ss = []
    fs = []
    es = []
    #Initialize the iterated state
    f = mp.matrix((f0))
    s = s0

    ss.append(s)
    fs.append(f)
    es.append(f - f)  # Initial step error is 0

    # Runge-Kutta45 coefficient table for values and truncation errors
    Ak = rk45Ak()
    Ck = rk45Ck()  

    CHk = rk45CHk()
    #This is just Ck-CHk
    CTk = rk45CTk()
    Bkl = rk45Bkl()

    ks = [0, 0, 0, 0, 0, 0]
    # Fixed step iterations of Runge-Kutta saving the error and results
    
    def fdiv2(a,b,c):
        if b == mpmathify(0):
            if c == mpmathify(0):
                return a
            return mp.fdiv(a,c)
        return mp.fdiv(a,b)

    ds = ds_max
    while s < s_final:
        if ds > ds_max:
            ds = ds_max
        if s_final - s < ds:
            ds = s_final - s
        #print(ds)
        delta = None

        args_s = s + ds * Ak

        truncation_error = None

        delta = None

        # Calculate the step change in the function
        # and the error estimate

        for k in range(6):
            fa = f
            for i in range(0, k):
                fa = fa + ks[i] * Bkl[k, i]

            ks[k] = ds * dfds(args_s[k], fa)

            if k == 0:
                delta = CHk[0] * ks[0]
                truncation_error = CTk[0] * ks[0]
            else:
                truncation_error = truncation_error + CTk[k] * ks[k]
                delta = delta + CHk[k] * ks[k]
        truncation_error[0] = mpmathify(0.0)
        err = max(ov(mp.fabs,(ov(fdiv2,truncation_error,ov(mp.fabs,f), delta))))
        if err   > step_tol and err != mpmathify(0.0):

            ds = 0.9 * ds * (step_tol/err )**(1/5) 
            print("shrink", err, ds/ s_final)
            continue
            
        f = f + delta
        s +=  ds
#Calculate ds based on the maxmimum relative error in the vector truncation_error/f

        if  max(ov(mp.fabs,(ov(fdiv2,truncation_error,ov(mp.fabs,f), delta)))) != mpmathify(0.0):
            ds = 0.9 * ds * (step_tol/max(ov(mp.fabs,(ov(fdiv2,truncation_error,f, delta)))))**(1/5) 

        es.append(truncation_error)
        ss.append(s)
        fs.append(f)

    return ss, fs, es


#A version of the RK89_fixed that uses a variable step size to reach the desired tolerance.
def mp_RKF89_adaptive(f0, s0, s_final, dfds, step_tol, ds_max):
    #I
    ss = []
    fs = []
    es = []
    #Initialize the iterated state
    f = mp.matrix((f0))
    s = s0
    print("STARTING!(IC): ", f)

    ss.append(s)
    fs.append(f)
    es.append(f - f)  # Initial step error is 0

    # Runge-Kutta89 coefficient table for values and truncation errors
    Ak = rk89Ak()
    #Error is given by Fehlberg more explicitly, doesnt require all the C coefficients
    CHk = rk89CHk()
    
    Bkl = rk89Bkl()

    ks = [0.0] * 17
    # Fixed step iterations of Runge-Kutta saving the error and results
    
    def fdiv2(a,b,c):
        if b == mpmathify(0):
            if c == mpmathify(0):
                return a
            return mp.fdiv(a,c)
        return mp.fdiv(a,b)

    ds = ds_max
    while s < s_final - ds_max:
        if s_final - s < ds:
            ds = s_final - s
        #print(ds, f)
        if ds > ds_max:
            ds = ds_max
        delta = None

        args_s = s + ds * Ak

        truncation_error = None

        delta = None

        # Calculate the step change in the function
        # and the error estimate

        for k in range(17):
            fa = f
            for i in range(0, k):
                fa = fa + ks[i] * Bkl[k, i]

            ks[k] = ds * dfds(args_s[k], fa)
            if k in [0,14,15,16]:
                if k == 0:
                    truncation_error = CHk[14] * ks[k]
                elif k == 14:
                    truncation_error = truncation_error + CHk[14] * ks[k]
                else:
                    truncation_error = truncation_error - CHk[14] * ks[k]

            if k == 0:
                delta = CHk[0] * ks[0]
            else:
                delta = delta + CHk[k] * ks[k]

        err = max(ov(mp.fabs,(ov(fdiv2,truncation_error,ov(mp.fabs,f), delta))))
        if err   > step_tol and err != mpmathify(0.0):
            ds = 0.9 * ds * (step_tol/err )**(1/9) 
            print("shrink", err, ds/ s_final)
            continue
            
        f = f + delta
        s +=  ds
#Calculate ds based on the maxmimum relative error in the vector truncation_error/f

        if  err != mpmathify(0.0):
            ds = 0.9 * ds * (step_tol/err)**(1/9) 

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


#Two refactored bending functions, one sets the side force, the oter sets the initial moment.
def bend_theta_with_m0(grid, hspline, thickness = 1, E=1, Fweight = mpmathify(1), Fside=mpmathify(0), theta0=1, tol=0.001,  use89 = False):
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    min_exponent = -12000

    onesmatrix = mp.matrix([1] * len(grid))

    #Use the spline of the geometry to be compatible with RK intermediate sampling. Assume a rectangular cross section although other cross sections are also trivial.
    IS = hspline
    def I_spline(x):
        #print(IS(x))
        return (IS(x)*2)**3 * thickness /12
    
    F1 = Fweight
    F2 = mpmathify(Fside)
    def dM_ds(t, Fco):
        return F1 * mp.sin(t) + Fco * mp.cos(t)


    def dt_ds(s, M):

        return M / E / I_spline(s)

    def df_ds(s, f):

        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0)])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    #Set the initial moment, or sideforce to a very small number, then the system is approximately linear and we can directly scale the initial condition to reach the desired bending angle.
    f0 = mp.matrix([mpmathify("1E" +  str(min_exponent)), mpmathify(0), Fside])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    f0 = mp.matrix(
        [mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1], mpmathify(0), Fside]
    )  # mpmathify()/F[len(F) - 1][1]
    if theta0 > 0.1:
        f0[0] = f0[0] /theta0 * 0.1

    #Define a function compatible with the default root finding. Use anderson as its fast and has convergence guarentee of regula falsi method
    def shot_function(x):
        ic = [0,0,Fside]
        ic[0] = x
        S, F, Es = bend(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        return F[-1][1] - theta0
    print(f0[0], "IC, M(0)")
    f0[0] = mp.findroot(shot_function, (f0[0]/64, f0[0] * mpmathify(64)), solver="anderson", tol=tol, verbose=True, verify=False)

    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    print(f0, "IC!")
    return S, F, Es

def bend_theta_y(grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), y0 = 1, theta0=1, tol=0.001, use89=False):
    #Estimate m0,Fs to acheive both y and heta0 on their own. This can help find bounds for the 2d optimization problem.

    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    min_exponent = -1200
    # Useful for shorthand calculation since we dont have total numpy freedom with mpmath library
    onesmatrix = mp.matrix([1] * len(grid))

    #Use the spline of the geometry to be compatible with RK intermediate sampling. Assume a rectangular cross section although other cross sections are also trivial.
    IS = hspline
    def I_spline(x):
        #print(IS(x))
        return (IS(x)*2)**3 * thickness /12

   


    def dM_ds(t, Fco):
        return Fweight * mp.sin(t) + Fco * mp.cos(t)


    def dt_ds(s, M):

        return M / E / I_spline(s)
    
    def dy_ds( theta):
        return mp.sin(theta)

    def df_ds(s, f):

        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0), dy_ds(f[1])])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    #Set the initial moment, or sideforce to a very small number, then the system is approximately linear and we can directly scale the initial condition to reach the desired bending angle.
    mat = [[0,0],[0,0]]
    #Matrix entries for small initial moment
    f0 = mp.matrix([mpmathify("1E" +  str(min_exponent)), mpmathify(0), mpmathify(0), mpmathify(0)])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    mat[0][1] = F[-1][3]/mpmathify("1E" +  str(min_exponent))
    mat[1][1] = F[-1][1]/mpmathify("1E" +  str(min_exponent))
    #Matrix entries for small side force
    f0 = mp.matrix([mpmathify(0), mpmathify(0),mpmathify("1E" + str(min_exponent)), mpmathify(0)])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    mat[0][0] = F[-1][3]/mpmathify("1E" +  str(min_exponent))
    mat[1][0] = F[-1][1]/mpmathify("1E" +  str(min_exponent))
    print("mat", mat[1][0]/mat[0][0], mat[1][1]/mat[0][1])
    for r in mat:
        print(r)
    tt = theta0
    yt = y0
    if theta0 > 0.1:
        tt = 0.1
    if yt/ L > 0.1:
        yt = 0.1*L
    res = ((mp.matrix(mat))**-1) * mp.matrix([[yt],[tt]])
    print(res, "res")
    #f0 = mp.matrix([res[1], mpmathify(0), res[0], mpmathify(0)])
    #S, F, Es = bend(
    #    f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    #)
    print(res,"Initial conditions decided, Fs,M0")
    #print(S[-1], F[-1][1],F[-1][3], "theta, y")
    def residuals(x):
        Fs, M0 = x[0],x[1]
        f0 = mp.matrix([M0, mpmathify(0), Fs, mpmathify(0)])
        S, F, Es = bend(f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0])
        theta_end = F[-1][1]
        y_end = F[-1][3]
        return ((y_end - y0)/grid[len(grid) - 1])**2 + (theta_end - theta0)**2
    # Newton shooting
    Fs_guess, M0_guess = res[0], res[1]
    Fss =[]
    M0ss =[]
    Ess = []

    for i in range(10):
        for j in range(10):
            Fs_sol, M0_sol = Fs_guess + (i-5) * Fs_guess / 1, M0_guess + (j-5) * M0_guess / 1
            f0 = mp.matrix([M0_sol, mpmathify(0), Fs_sol, mpmathify(0)])
            S, F, Es = bend(f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0])
            theta_end = F[-1][1]
            y_end = F[-1][3]
            res = ((y_end - y0)/grid[len(grid) - 1])**2 + (theta_end - theta0)**2
            
            Fss.append(Fs_sol)
            M0ss.append(M0_sol)
            Ess.append(res)
            print(res, type(res))
    print(Es)
    # Convert to numpy arrays
    Fss_arr = np.array([float(x/ Fss[0]) for x in Fss], dtype=float)
    M0s_arr = np.array([float(x/ M0ss[0]) for x in M0ss], dtype=float)
    Es_arr  = np.array([float(x) for x in Ess], dtype=float)
#"""ERROR TODO
#      File "C:\Users\Ben\Documents\PleaseGitHub\BeamBending\functions_bending_schreyer_adaptive89.py", line 388, in bend_theta_y
#    Es_arr  = np.array(Es, dtype=float)
#ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1773,) + inhomogeneous part."""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Option 1: scatter plot of raw data
    sc = ax.scatter(np.array(Fss_arr), np.array(M0s_arr), np.array(Es_arr), c=Es_arr, cmap='viridis', s=50)
    fig.colorbar(sc, ax=ax, label='Residual E')

    # Option 2: surface plot (requires gridded data)
    # If your points are on a grid:
    # Fs_unique = np.unique(Fss_arr)
    # M0_unique = np.unique(M0s_arr)
    # Fs_grid, M0_grid = np.meshgrid(Fs_unique, M0_unique)
    # E_grid = Es_arr.reshape(Fs_grid.shape)
    # surf = ax.plot_surface(Fs_grid, M0_grid, E_grid, cmap='viridis')
    # fig.colorbar(surf, ax=ax, label='Residual E')

    ax.set_xlabel("Fs")
    ax.set_ylabel("M0")
    ax.set_zlabel("Residual E")
    ax.set_title("Residual Surface over Fs, M0")

    plt.tight_layout()
    plt.show()
    

    print("GD done:", Fs_sol, M0_sol, "objective =", f_star, info)
    # sol is a tuple-like mpmath matrix/vector
    #Fs_sol, M0_sol = sol[0], sol[1]

    # Final run with solved initial conditions
    f0 = mp.matrix([M0_sol, mpmathify(0), Fs_sol, mpmathify(0)])
    S, F, Es = bend(f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0])

    # Print results
    print(f"Solved Fs = {Fs_sol}, M0 = {M0_sol}")
    print(f"End theta = {F[-1][1]}, End y = {F[-1][3]}")

    return S, F, Es


def bend_theta_with_Fside(
    grid, hspline, order=4, E=1, thickness=1, Fweight=mpmathify(1), M0=mpmathify(0), theta0=1, tol=0.001, use89=False
):
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    min_exponent = -12000
    # Useful for shorthand calculation since we dont have total numpy freedom with mpmath library
    onesmatrix = mp.matrix([1] * len(grid))

    #Use the spline of the geometry to be compatible with RK intermediate sampling. Assume a rectangular cross section although other cross sections are also trivial.
    IS = hspline
    def I_spline(x):
        #print(IS(x))
        return (IS(x)*2)**3 * thickness /12
    F1 = Fweight
    M0 = mpmathify(M0)



    def dM_ds(t, Fco):
        return F1 * mp.sin(t) + Fco * mp.cos(t)


    def dt_ds(s, M):

        return M / E / I_spline(s)

    def df_ds(s, f):

        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0)])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    f0 = mp.matrix([M0, mpmathify(0),mpmathify("1E" + str(min_exponent)),0])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    f0 = mp.matrix(
        [M0, mpmathify(0), mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1],0]
    )  # mpmathify()/F[len(F) - 1][1]
    if theta0 > 0.1:
        f0[2] = f0[2] /theta0 * 0.1
    def shot_function(x):
        ic = [M0,0,0]
        ic[2] = x
        S, F, Es = bend(
        mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print(F[-1][1])
        return F[-1][1] - theta0
    print(f0[2], "IC, Fs")
    f0[2] = mp.findroot(shot_function, (f0[2]/64, f0[2] * mpmathify(64)), solver="anderson", tol=tol, verbose=True, verify=False)

    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    print(f0, "IC!")
    return S, F, Es

# Bending calculation generalized for a moment M0 or a side force Fcos
def bend_samples(
    grid, hspline, order=4, E=1, Fsin=mpmathify(0), Fcos=False, M0 = None, theta0=1, tol=0.001, T = 1, use89 = False
):
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    min_exponent = -12000
    # Useful for shorthand calculation since we dont have total numpy freedom with mpmath library
    onesmatrix = mp.matrix([1] * len(grid))

    #Use the spline of the geometry to be compatible with RK intermediate sampling. Assume a rectangular cross section although other cross sections are also trivial.
    IS = hspline
    def I_spline(x):
        #print(IS(x))
        return (IS(x)*2)**3 * T /12

    # Is the sine term present?
    Fs = not (Fsin == mpmathify(0))
    # Is the bending angle small enough to use linear approximation?

    theta_is_small = mp.fabs(theta0) ** 2 / 2 < tol

    F1 = Fsin
    F2 = mpmathify(0)



    def dM_ds(t, Fco):
        return F1 * mp.sin(t) + Fco * mp.cos(t)


    def dt_ds(s, M):

        return M / E / I_spline(s)
    
    def dy_ds( theta):
        return mp.sin(theta)

    def df_ds(s, f):

        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0), dy_ds(f[1])])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    #Set the initial moment, or sideforce to a very small number, then the system is approximately linear and we can directly scale the initial condition to reach the desired bending angle.
    if not Fcos:
        f0 = mp.matrix([mpmathify("1E" +  str(min_exponent)), mpmathify(0), mpmathify(0), mpmathify(0)])
        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1], mpmathify(0), mpmathify(0), mpmathify(0  )]
        )  # mpmathify()/F[len(F) - 1][1]
        if theta0 > 0.1:
            f0[0] = f0[0] /theta0 * 0.1
    else:
        f0 = mp.matrix([mpmathify(0), mpmathify(0),mpmathify("1E" + str(min_exponent)), mpmathify(0)])
        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify(0), mpmathify(0), mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1], mpmathify(0)]
        )  # mpmathify()/F[len(F) - 1][1]
        if theta0 > 0.1:
            f0[2] = f0[2] /theta0 * 0.1
    print("SIMPLIFIED CODE, NO GUESSING")

    #############
    ###"Shooting"
    #############
    #Shoot the Fcos or initial moment depending on the type of bending. We need to do shooting incase the angle is large and nonlinearity makes shooting nontrivial.
    if not Fcos:
        #Define a function compatible with the default root finding. Use anderson as its fast and has convergence guarentee of regula falsi method
        def shot_function(x):
            ic = [0,0,0,0]
            ic[0] = x
            S, F, Es = bend(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
            )
            return F[-1][1] - theta0
        print(f0[0], "IC, M(0)")
        f0[0] = mp.findroot(shot_function, (f0[0]/64, f0[0] * mpmathify(64)), solver="anderson", tol=tol, verbose=True, verify=False)

        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print(f0, Fs, Fcos)
        return S, F, Es
    else:
        def shot_function(x):
            ic = [0,0,0,0]
            ic[2] = x
            S, F, Es = bend(
            mp.matrix(ic), s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
            )
            print(F[-1][1])
            return F[-1][1] - theta0
        print(f0[2], "IC, Fs")
        f0[2] = mp.findroot(shot_function, (f0[2]/64, f0[2] * mpmathify(64)), solver="anderson", tol=tol, verbose=True, verify=False)

        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        print(f0, Fs, Fcos, "IC!")
        return S, F, Es
    # Cosine only, shoots the F2 term which is the force coefficient on the cosine force, also for the general case or both terms significant
    

print("Main!")
L = 0.1
wh = 0.001
def hsc(s):
    return 0.01 # + (s - L/2)**2 * 4

def h(s):
    return wh * hsc(s)
s_eval = mp.matrix(np.linspace(0,L,int(200),endpoint = True))
bend_theta_y(s_eval, h, thickness=mpmathify(0.001), E=mpmathify(10**10), Fweight=mpmathify(1), y0 = mpmathify(0.00005), theta0=mpmathify(0.001), tol=0.000001, use89=True)