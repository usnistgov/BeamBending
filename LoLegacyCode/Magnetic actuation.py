import matplotlib.pyplot as plt
import numpy as np


#%% Variables and Parameters

mu0 = 4 * np.pi * 1e-7 # Tm/A
N = 1000
I = 50e-3 # A
L = 0.01 # m, length of the wound coil stack
chi = 1e-6 # susceptibility fused silica
l = 0.005 # m, exposed material length
w = 0.01 # m, exposed material width
h = 0.005 # m, exposed material height
V = l * w * h
r = 25.4e-3 # m, bob radius
kb = 1.4e-23 # J/K
f = 0.07 # Hz
T = 293 # K
k = 1e-5 # Nm/rad
Q = 1e2 # mechanical quality factor
rho_C = 1.7e-8 # Ohm / m
r_C = 0.008/2*25.4e-3 # m, magnet wire diameter gauge 32
d_F = 28e-3 # m, mean diameter of coil former

#%% Calculation of magnetic moment for pendulum excitation

B = mu0 * N * I / L 
print('B=', B)
F = chi * B**2 * V / mu0 / L
print('F=', F)
M = 2 * r * F
print('M=', M)

S = np.sqrt((4 * kb * T * k) / Q / (2 * np.pi * f))
print('S=', S)
angle = M / k
print('angle=', angle)

#%% Resistance of coil and compliance voltage

A_C = np.pi * r_C**2
l_C = d_F * np.pi * N
print('l_C=', l_C)
R = l_C * rho_C / A_C * 2 # for two coils
print('R=', R)
V = R * I
print('V=', V)