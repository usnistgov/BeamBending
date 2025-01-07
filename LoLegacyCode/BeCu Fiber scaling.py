import matplotlib.pyplot as plt
import numpy as np


#%% Yield strength round fiber
m = 0.01 #np.linspace(0.01, 0.03, 3) # kg
g = 9.81 # N/kg
sigma_y = 750 # MPa
Sf = 1.2 # Safety factor to yield
rmin= np.sqrt((m*g)/(np.pi*sigma_y/Sf))/1000 # in meter
print(rmin)
rmax = 4*rmin
#rmin=18e-6/2
#rmax=125e-6/2



#%%


E = 400000e6 # Pa
nu = 0.3
G = E / (2 * (1 + nu)) # Pa
l = 0.02 # fiber length in m
D = np.linspace(rmin*2, rmax*2) # fiber diameter in m
rbob = 2/2*25.4/1000 #0.02 # m
print("rbob=",rbob) # d is half diameter of bob. so d is r of the bob.

kappa = (G * np.pi * (D**4/32)) / l
T_dum = 2 * np.pi * np.sqrt((m * rbob**2) / kappa) # m instead of m for MOI of disk instead of dumbell
T_disc = 2 * np.pi * np.sqrt((0.5*m * rbob**2) / kappa) # 0.5*m instead of m for MOI of disk instead of dumbell
print(kappa)
print(G/1e9)

fig, ax = plt.subplots(1)
ax.plot(D*1e6, T_dum, label='MOI according to dumbell config')
ax.plot(D*1e6, T_disc, label='MOI according to disc config')
ax.set_ylabel(r'$T / \mathrm{s}$')
ax.set_xlabel(r'$D / \mathrm{\mu m}$')
ax.legend()

#print(T[-1])

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


