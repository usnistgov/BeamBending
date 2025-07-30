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
    #This is just Ck-CHk
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
    Ak = mp.matrix(
        [
            mpmathify(" 0.0"),
        mpmathify(" 0.44368940376498183109599404281370"),
        mpmathify(" 0.66553410564747274664399106422055"),
        mpmathify(" 0.99830115847120911996598659633083"),
        mpmathify(" 0.3155"),
        mpmathify(" 0.50544100948169068626516126737384"),
        mpmathify(" 0.17142857142857142857142857142857"),
        mpmathify(" 0.82857142857142857142857142857143"),
        mpmathify(" 0.66543966121011562534953769255586"),
        mpmathify(" 0.24878317968062652069722274560771"),
        mpmathify(" 0.1090"),
        mpmathify(" 0.8910"),
        mpmathify(" 0.3995"),
        mpmathify(" 0.6005"),
        mpmathify(" 1.0"),
        mpmathify("0.0"),
        mpmathify(" 1.0"),
        ]
    )
    #Error is given by Fehlberg more explicitly, doesnt require all the C coefficients
    CHk = mp.matrix(
        [
            mpmathify("0.32256083500216249913612900960247e-1"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.0"),
            mpmathify("0.25983725283715403018887023171963"),
            mpmathify("0.92847805996577027788063714302190e-1"),
            mpmathify("0.16452339514764342891647731842800"),
            mpmathify("0.17665951637860074367084298397547"),
            mpmathify("0.23920102320352759374108933320941"),
            mpmathify("0.39484274604202853746752118829325e-2"),
            mpmathify("0.30726495475860640406368305522124e-1"),
            mpmathify("0.0"),
            mpmathify("0.0"),
   
        ]
    )
    
    Bkl = mp.matrix(
        [
[mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify(" 0.44368940376498183109599404281370"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.16638352641186818666099776605514"),
mpmathify("  0.49915057923560455998299329816541"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.24957528961780227999149664908271"),
mpmathify(" 0.0"),
mpmathify("  0.74872586885340683997448994724812"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.20661891163400602426556710393185"),
mpmathify(" 0.0"),
mpmathify("  0.17707880377986347040380997288319"),
mpmathify("  -0.68197715413869494669377076815048e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.10927823152666408227903890926157"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.40215962642367995421990563690087e-2"),
mpmathify("  0.39214118169078980444392330174325"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.98899281409164665304844765434355e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.35138370227963966951204487356703e-2"),
mpmathify("  0.12476099983160016621520625872489"),
mpmathify("  -0.55745546834989799643742901466348e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.36806865286242203724153101080691"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.22273897469476007645024020944166e+1"),
mpmathify("  0.13742908256702910729565691245744e+1"),
mpmathify("  0.20497390027111603002159354092206e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.45467962641347150077351950603349e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.32542131701589147114677469648853"),
mpmathify("  0.28476660138527908888182420573687"),
mpmathify("  0.97837801675979152435868397271099e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.60842071062622057051094145205182e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.21184565744037007526325275251206e-1"),
mpmathify("  0.19596557266170831957464490662983"),
mpmathify("  -0.42742640364817603675144835342899e-2"),
mpmathify("  0.17434365736814911965323452558189e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.54059783296931917365785724111182e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.11029825597828926530283127648228"),
mpmathify("  -0.12565008520072556414147763782250e-2"),
mpmathify("  0.36790043477581460136384043566339e-2"),
mpmathify("  -0.57780542770972073040840628571866e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.12732477068667114646645181799160"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.11448805006396105323658875721817"),
mpmathify("  0.28773020709697992776202201849198"),
mpmathify("  0.50945379459611363153735885079465"),
mpmathify("  -0.14799682244372575900242144449640"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.36526793876616740535848544394333e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  0.81629896012318919777819421247030e-1"),
mpmathify("  -0.38607735635693506490517694343215"),
mpmathify("  0.30862242924605106450474166025206e-1"),
mpmathify("  -0.58077254528320602815829374733518e-1"),
mpmathify("  0.33598659328884971493143451362322"),
mpmathify("  0.41066880401949958613549622786417"),
mpmathify("  -0.11840245972355985520633156154536e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("   -0.12375357921245143254979096135669e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("   -0.24430768551354785358734861366763e+2"),
mpmathify("  0.54779568932778656050436528991173"),
mpmathify("   -0.44413863533413246374959896569346e+1"),
mpmathify("  0.10013104813713266094792617851022e+2"),
mpmathify("   -0.14995773102051758447170985073142e+2"),
mpmathify("  0.58946948523217013620824539651427e+1"),
mpmathify("  0.17380377503428984877616857440542e+1"),
mpmathify("  0.27512330693166730263758622860276e+2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.35260859388334522700502958875588"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.18396103144848270375044198988231"),
mpmathify("  -0.65570189449741645138006879985251"),
mpmathify("  -0.39086144880439863435025520241310"),
mpmathify("  0.26794646712850022936584423271209"),
mpmathify("  -0.10383022991382490865769858507427e+1"),
mpmathify("  0.16672327324258671664727346168501e+1"),
mpmathify("  0.49551925855315977067732967071441"),
mpmathify("  0.11394001132397063228586738141784e+1"),
mpmathify("  0.51336696424658613688199097191534e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  0.10464847340614810391873002406755e-2"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.67163886844990282237778446178020e-2"),
mpmathify("  0.81828762189425021265330065248999e-2"),
mpmathify("  -0.42640342864483347277142138087561e-2"),
mpmathify("  0.28009029474168936545976331153703e-3"),
mpmathify("  -0.87835333876238676639057813145633e-2"),
mpmathify("  0.10254505110825558084217769664009e-1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
],
[mpmathify("  -0.13536550786174067080442168889966e+1"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify(" 0.0"),
mpmathify("  -0.18396103144848270375044198988231"),
mpmathify("  -0.65570189449741645138006879985251"),
mpmathify("  -0.39086144880439863435025520241310"),
mpmathify("  0.27466285581299925758962207732989"),
mpmathify("  -0.10464851753571915887035188572676e+1"),
mpmathify("  0.16714967667123155012004488306588e+1"),
mpmathify("  0.49523916825841808131186990740287"),
mpmathify("  0.11481836466273301905225795954930e+1"),
mpmathify("  0.41082191313833055603981327527525e-1"),
mpmathify(" 0.0"),
mpmathify("   1.0"),
mpmathify(" 0.0"),
],
]
    )

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

    f0 = mp.matrix([M0, mpmathify(0),mpmathify("1E" + str(min_exponent))])
    S, F, Es = bend(
        f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
    )
    f0 = mp.matrix(
        [M0, mpmathify(0), mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1]]
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

    def df_ds(s, f):

        return mp.matrix([dM_ds(f[1], f[2]), dt_ds(s, f[0]), mpmathify(0)])

    f0 = None
    # Anytime the cosine term is not negligible (a side force is present)

    s0 = grid[0]

    #Set the initial moment, or sideforce to a very small number, then the system is approximately linear and we can directly scale the initial condition to reach the desired bending angle.
    if not Fcos:
        f0 = mp.matrix([mpmathify("1E" +  str(min_exponent)), mpmathify(0), mpmathify(0)])
        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1], mpmathify(0), mpmathify(0)]
        )  # mpmathify()/F[len(F) - 1][1]
        if theta0 > 0.1:
            f0[0] = f0[0] /theta0 * 0.1
    else:
        f0 = mp.matrix([mpmathify(0), mpmathify(0),mpmathify("1E" + str(min_exponent))])
        S, F, Es = bend(
            f0, s0, grid[len(grid) - 1], df_ds, tol, grid[1] - grid[0]
        )
        f0 = mp.matrix(
            [mpmathify(0), mpmathify(0), mpmathify("1E" + str(min_exponent)) * theta0 / F[len(F) - 1][1]]
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
            ic = [0,0,0]
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
            ic = [0,0,0]
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
    

