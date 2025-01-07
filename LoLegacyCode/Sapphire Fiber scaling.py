import matplotlib.pyplot as plt
import numpy as np
import math
from mpmath import *



#%% monofilar
m = 0.04 #np.linspace(0.01, 0.03, 3) # kg
g = 9.81 # N/kg
sigma_y = 3000 # MPa
Sf = 2 # Safety factor to yield
rmin= np.sqrt((m*g)/(np.pi*sigma_y/Sf))/1000 # in meter
rmax = 3*rmin
print(rmin*2, rmax*2)
# rmin=125e-6/2
# rmax=150e-6/2



E = 72000e6 # Pa, worst case from matweb https://www.matweb.com/search/datasheet.aspx?matguid=a188c3ef359945f7a6c04b9aadb0f42e&ckck=1
nu = 0.17
G = E / (2 * (1 + nu)) # Pa
l = 0.005 # fiber length in m
D = np.linspace(rmin*2, rmax*2) # fiber diameter in m
rbob = 1/2*25.4/1000 #0.02 # m
print("rbob=",rbob) # d is half diameter of bob. so d is r of the bob.

kappa_el = (G * np.pi * (D**4/32)) / l
T_dum = 2 * np.pi * np.sqrt((m * rbob**2) / kappa_el) # m instead of m for MOI of disk instead of dumbell
T_disc = 2 * np.pi * np.sqrt((0.5*m * rbob**2) / kappa_el) # 0.5*m instead of m for MOI of disk instead of dumbell
print(T_disc[-1])
print(kappa_el)

A = (D**2/4) * np.pi
kappa_axial = E * A / l
w_axial = np.sqrt(kappa_axial / m)
print('f bounce:', w_axial / 2 / np.pi)

normal_stress = m * g / A
print('normal_stress / MPa:', normal_stress * 1e-6)

fig, ax = plt.subplots(1)
ax.plot(D*1e6, T_dum, label='MOI according to dumbell config')
ax.plot(D*1e6, T_disc, label='MOI according to disc config')
ax.set_ylabel(r'Period $/ \mathrm{s}$')
ax.set_xlabel(r'Diameter $/ \mathrm{\mu m}$')
ax.legend()

# fig, ax = plt.subplots(1)
# ax1 = ax.twinx()
# ax.plot(D*1e6, kappa, 'r', label = '$\kappa$ for $l = 10$ mm')
# ax1.plot(D*1e6, T, 'k', label = '$T$ for $l = 10$ mm')
# ax.set_ylabel(r'$\kappa / \mathrm{Nm/rad}$')
# ax1.set_ylabel(r'$T / \mathrm{s}$')|
# ax.set_xlabel(r'$D / \mathrm{\mu m}$')
# ax.legend(loc = 'upper center')
# ax1.legend(loc = 'center')

#print(T[-1])


#%% n-filar
m = 0.06 #np.linspace(0.01, 0.03, 3) # kg
g = 9.81 # N/kg
sigma_y = 1400 # MPa
Sf = 2 # Safety factor to yield
n = 2 # number of fibers
rmin= np.sqrt((m*g)/(np.pi*sigma_y/Sf))/1000/np.sqrt(n) # in meter, sqrt(n) for number of fibers
rmin=rmin*1.5
rmax = 2*rmin
print(rmin*2, rmax*2)
# rmin=125e-6/2
# rmax=150e-6/2

E = 128000e6 # Pa, worst case from matweb https://www.matweb.com/search/datasheet.aspx?matguid=a188c3ef359945f7a6c04b9aadb0f42e&ckck=1
nu = 0.3
G = E / (2 * (1 + nu)) # Pa
print('G', G)
l = 100e-3 # fiber length in m
D = np.linspace(rmin*2, rmax*2) # fiber diameter in m
print("fiber diameter", D)
rbob = 8/2*25.4/1000 #0.02 # m
print("rbob=",rbob) # d is half diameter of bob. so d is r of the bob.♣
b = 4000e-6 # fiber separation in meter

# a part of the stiffness is bending. Constant cross section: Clive equation
L = 0.0001e-3 # length of end section that bends in m
W = m * g / (n)
Ds = 20e-6 # end rod diameter in m
H = np.pi * ((Ds/2)**4/4)
alpha = W / (E * H)
deamp = b / (2 * l)  # deamplification of bending angle by fiber length and b   
rf = 1/(2*alpha)*(1.0/np.tanh(alpha * L) - alpha * L / np.sinh(alpha * L)**2)
scaling =  deamp# / rf * b/2 # amplification of bending angle by fiber length and b, not sure about the "/ rf * b/2"
kappa_el_bend = scaling * W/(2*alpha)*(1.0/np.tanh(alpha * L) + alpha * L / np.sinh(alpha * L)**2) * 2 * n
kappa_g_bend = scaling * W/(2*alpha)*(1.0/np.tanh(alpha * L) - alpha * L / np.sinh(alpha * L)**2) * 2 * n

kappa_el = (G * np.pi * (D**4/32)) / l * n # * n for number of fibers
kappa_el_tot = kappa_el + kappa_el_bend
kappa_g = m * g * b**2 / l
kappa_g_tot = kappa_g + kappa_g_bend
T_dum = 2 * np.pi * np.sqrt((m * rbob**2) / (kappa_el_tot + kappa_g_tot)) # m instead of m for MOI of disk instead of dumbell
T_disc = 2 * np.pi * np.sqrt((0.5*m * rbob**2) / (kappa_el_tot + kappa_g_tot)) # 0.5*m instead of m for MOI of disk instead of dumbell

dil_factor = 1 + kappa_g_tot / kappa_el_tot
print(T_disc[-1])
print("kappael", kappa_el)
print("kappag",kappa_g_tot)

A = (D**2/4) * np.pi * n
kappa_axial = E * A / l
w_axial = np.sqrt(kappa_axial / m)
print('f bounce:', w_axial / 2 / np.pi)

normal_stress = m * g / A
print('normal_stress / MPa:', normal_stress * 1e-6)

fig, ax = plt.subplots(1)
ax1 = ax.twinx()
ax1.plot(D*1e6, dil_factor, 'r')
ax.plot(D*1e6, T_dum, label='MOI according to dumbell config')
ax.plot(D*1e6, T_disc, label='MOI according to disc config')
ax.set_ylabel(r'Period $/ \mathrm{s}$')
ax.set_xlabel(r'Diameter $/ \mathrm{\mu m}$')
ax.legend()

# fig, ax = plt.subplots(1)
# ax1 = ax.twinx()
# ax.plot(D*1e6, kappa, 'r', label = '$\kappa$ for $l = 10$ mm')
# ax1.plot(D*1e6, T, 'k', label = '$T$ for $l = 10$ mm')
# ax.set_ylabel(r'$\kappa / \mathrm{Nm/rad}$')
# ax1.set_ylabel(r'$T / \mathrm{s}$')
# ax.set_xlabel(r'$D / \mathrm{\mu m}$')
# ax.legend(loc = 'upper center')
# ax1.legend(loc = 'center')

#print(T[-1])

#%% tryout cells
L = np.linspace(0.0001e-3, 0.0002e-3)
kappa_el_bend_list = []
rf_list = []
for i in L:
    kappa_el_bend = W/(2*alpha)*(1.0/np.tanh(alpha * i) + alpha * i / np.sinh(alpha * i)**2)
    rf = 1/(2*alpha)*(1.0/np.tanh(alpha * i) - alpha * i / np.sinh(alpha * i)**2)
    scaling =  deamp / rf * b/2
    kappa_el_bend_list.append(kappa_el_bend * scaling)
    rf_list.append(rf)

fig, ax = plt.subplots(1)

ax.plot(L, kappa_el_bend_list)
print(rf_list)

#%%


l = np.linspace(0.02, 0.2) # m
D = 0.05e-3 # m

kappa = (G * np.pi * (D**4/32)) / l
T = 2 * np.pi * np.sqrt((m * d**2) / kappa)

fig, ax = plt.subplots(1)
ax1 = ax.twinx()
ax.plot(l*1e2, kappa, 'r', label = '$\kappa$ for $D = 0.05$ mm')
ax1.plot(l*1e2, T, 'k', label = '$T$ for $D = 0.05$ mm')
ax.set_ylabel(r'$\kappa / \mathrm{Nm/rad}$')
ax.set_xlabel(r'$l / \mathrm{cm}$')
ax.legend(loc = 'upper center')
ax1.legend(loc = 'right')

print(T[0])

#%%


l = 0.05 # m
D = 0.05e-3 # m

kappa = (G * np.pi * (D**4/32)) / l
T = 2 * np.pi * np.sqrt((m * d**2) / kappa)

print(T)

#%%


