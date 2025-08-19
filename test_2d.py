from functions_bending_schreyer_adaptive89 import *
import matplotlib.pyplot as plt
from mpmath import mpmathify
import mpmath as mp
import numpy as np
import math
import time
from scipy.interpolate import CubicSpline
from rungekuttacoefficients import *


print("Main!")
L = 0.1
wh = 0.001
def hsc(s):
    return 0.1 # + (s - L/2)**2 * 4

def h(s):
    return wh * hsc(s)

alpap = np.sqrt(0.1/10**10/(h(0)**3 * 0.001/12))
s_eval = mp.matrix(np.linspace(0,L,int(200),endpoint = True))
#compare_coordinates_schemes(grid, hspline, thickness=1, E=1, Fweight=mpmathify(1), m0 = 1, Fs=1, tol=0.001, use89=False):
m0 =  0.0001#mpmathify("1E-12000")

bend_to_y_theta(s_eval, h, thickness=mpmathify(0.001), E=mpmathify(10**10), Fweight=mpmathify(0.1), y0 = L * 0.1, theta0 = 0.3, tol=0.0001, use89=True)
      