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
min_exponent = -12000
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
    print("LOOP 89")
    count = 0
    while s < s_final - ds_max:
        count +=1
        if count % 1000 == 0:
            print(count, s)
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
            #print("shrink", err, ds/ s_final)
            continue
            
        f = f + delta
        s +=  ds
#Calculate ds based on the maxmimum relative error in the vector truncation_error/f

        if  err != mpmathify(0.0):
            ds = 0.9 * ds * (step_tol/err)**(1/9) 

        es.append(truncation_error)
        ss.append(s)
        fs.append(f)
    print("DONE 89")
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

def bend_to_y_theta(grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), y0 = 1, theta0=1, tol=0.001, use89=False):
    L = grid[len(grid) - 1]
    a = a_parameter(grid, hspline, thickness, E, Fweight, tol, use89)
    atransform = a

    target = mp.matrix([[-1, L],[atransform, -1]]) * mp.matrix([[y0],[theta0]])



    guess = (alt_cor_response_matrix(a,  grid, hspline, thickness, E, Fweight, tol, use89)**-1) * target

    S, F, Es = bend_alt_cor(a, grid, hspline, thickness, E, Fweight, guess[1], guess[0], tol, use89)

    resu = mp.matrix([[F[-1][2]], [F[-1][1]]])
    print("\n\n\n\n",target, "TARGET", guess, "GUESS", resu, "RESULT","\n\n\n\n")
    def objective1(Fs, m0):
        S, F, Es = bend_alt_cor(a, grid, hspline, thickness, E, Fweight, m0, Fs, tol, use89)

        resu = mp.matrix([[F[-1][2]], [F[-1][1]]])
        return ((target[0] - resu[0])/L)

    def objective2(Fs, m0):
        S, F, Es = bend_alt_cor(a, grid, hspline, thickness, E, Fweight, m0, Fs, tol, use89)

        resu = mp.matrix([[F[-1][2]], [F[-1][1]]])
        return ((target[1] - resu[1]))
    Fss,M0ss, Ess = [],[],[]
    N, M = 5,5
    for i in range(N):
        for j in range(M):
            print(100*(M*i + j)/M/N, "% prog")
            Fs_sol = guess[0] * (1 + (i - N//2)/N*8 )
            M0_sol = guess[1] * (1 + (j - M//2)/M*8 )
            print(Fs_sol, M0_sol)
            res = objective1(Fs_sol,M0_sol)**2 + objective2(Fs_sol,M0_sol)**2
            Fss.append(Fs_sol)
            M0ss.append(M0_sol)
            Ess.append(res)
    #        print(res, type(res))
    #print(Es)
    # Convert to numpy arrays
    Fss_arr = np.array([float(x/ Fss[0]) for x in Fss], dtype=float)
    M0s_arr = np.array([float(x/ M0ss[0]) for x in M0ss], dtype=float)
    Es_arr  = np.array([float(x) for x in Ess], dtype=float)
#"""ERROR TODO
#     File "C:\Users\Ben\Documents\PleaseGitHub\BeamBending\functions_bending_schreyer_adaptive89.py", line 388, in bend_theta_y #   Es_arr  = np.array(Es, dtype=float)
#ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1773,) + inhomogeneous part."""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Option 1: scatter plot of raw data
    sc = ax.scatter(np.array(Fss_arr), np.array(M0s_arr), np.array(Es_arr), c=Es_arr, cmap='viridis', s=50)
    fig.colorbar(sc, ax=ax, label='Residual E')

    plt.show()


    solution = mp.findroot([objective1, objective2], (guess[0], guess[1]), verbose = True, verify = False)
    Fs_sol, m0_sol = solution
    print(Fs_sol, m0_sol)
    S, F, Es = bend_alt_cor(a, grid, hspline, thickness, E, Fweight, m0_sol, Fs_sol, tol, use89)

    print("\n\n\n\n",target, "TARGET",  mp.matrix([[F[-1][2]],[F[-1][1]]]), "RESULTopt","\n\n\n\n")


def alt_cor_response_matrix(atransform,grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), tol=0.001, use89=False):
    L = grid[len(grid) - 1]
    m0mini = mpmathify("1E" + str(min_exponent))
    Fsmini = mpmathify("1E" + str(min_exponent))

    #generate linear approximate responses
    Sm0, Fm0, Esm0 = bend_alt_cor(atransform, grid,hspline, thickness, E, Fweight, m0mini, 0, tol, use89)
    SFs, FFs, EFs = bend_alt_cor(atransform, grid,hspline, thickness, E, Fweight, 0, Fsmini, tol, use89)
    ret = mp.matrix([[FFs[-1][2]/Fsmini,Fm0[-1][2]/m0mini],[FFs[-1][1]/Fsmini,Fm0[-1][1]/m0mini]])
    print(ret[0,0] * ret[1,1] - ret[1,0] * ret[0,1], "DETERMINANT!")
    return ret

def bend_alt_cor(atransform,grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), m0 = 1, Fs=1, tol=0.001, use89=False):
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    
    IS = hspline
    def I_spline(x):
        #print(IS(x))
        return (IS(x)*2)**3 * thickness /12
    L = grid[len(grid) - 1]
    #define the system in this transform

    def thetat(A,B):
        return (A + atransform * B)/(atransform * L - 1)

    def dA_ds(A,B,M,s):
        return atransform * mp.sin(thetat(A,B)) - M / E / I_spline(s)
    
    def dB_ds(A,B,M,s):
        return -1 * mp.sin(thetat(A,B)) + L * M/E / I_spline(s)
    
    def dM_ds_tran(A,B, Fs):
        return Fweight * mp.sin(thetat(A,B)) + Fs * mp.cos(thetat(A,B))
    
    def df_ds_tran(s, f):
        #[dM, dA, dB, dFs]
        return mp.matrix([dM_ds_tran(f[1], f[2], f[3]), dA_ds(f[1],f[2],f[0],s), dB_ds(f[1],f[2],f[0],s), mpmathify(0)])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]


    f0 = mp.matrix([m0, mpmathify(0), mpmathify(0), Fs])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds_tran, tol, grid[1] - grid[0]
    )

    #transform back to y, theta coordinates
    #gp=[mp.matrix([[-1, L],[atransform, -1]])**-1 * mp.matrix([[F[i][2]],[F[i][1]]]) for i in range(len(F))]

    return S, F, Es


#generate the coordinate change parameter a to avoid roundoff error
def a_parameter(grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), tol=0.001, use89=False):
    #Estimate m0,Fs to acheive both y and heta0 on their own. This can help find bounds for the 2d optimization problem.
    L = grid[len(grid) - 1]
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    
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
    mato = [[0,0],[0,0]]
    #Matrix entries for small initial moment
    f0 = mp.matrix([mpmathify("1E" +  str(min_exponent)), mpmathify(0), mpmathify(0), mpmathify(0)])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    mato[0][1] = F[-1][3]/mpmathify("1E" +  str(min_exponent))
    mato[1][1] = F[-1][1]/mpmathify("1E" +  str(min_exponent))
    #Matrix entries for small side force
    f0 = mp.matrix([mpmathify(0), mpmathify(0),mpmathify("1E" + str(min_exponent)), mpmathify(0)])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    mato[0][0] = F[-1][3]/mpmathify("1E" +  str(min_exponent))
    mato[1][0] = F[-1][1]/mpmathify("1E" +  str(min_exponent))

    M = mato



    atransform = mato[1][0]/mato[0][0]
    return atransform

# Return S (the places along the fiber for which the bending angle is determined), F (list of vectors [M,theta,F2] where theta encodes the geometry of the bending, and Es is the errors
# Bending calculation for a zero moment zero theta initial flexure with "shape" Isamples
# grid: locations along the fiber the moment is sampled, Isamples: samples of the moment for bending along the fiber (same length as grid), order: not used, E: modulus of the material, Fsin: force coefficent for the sin term in bending, Fcos: boolean determining if a side force is present, theta0: intended bending angle, tol: tolerance for shooting


#Two refactored bending functions, one sets the side force, the oter sets the initial moment.
def bend_theta_with_m0(grid, hspline, thickness = 1, E=1, Fweight = mpmathify(1), Fside=mpmathify(0), theta0=1, tol=0.001,  use89 = False):
    bend = mp_RKF45_adaptive
    if use89:
        bend = mp_RKF89_adaptive
    

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
