#Generated example script to calculate bending
#Benjamin Schreyer
#stephan.schlamminger@nist.gov
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import make_interp_spline

from functions_bending_schreyer_adaptive89 import bend_samples, integrate_xz

# ---- Parameters ----
L = 0.05               # Length (m)
T = 20e-6              # Thickness (m)
w0 = 0.5e-3            # Mean half-width (m)
amp = 0.3e-3           # Amplitude of sinusoid (m)
E = 200e9              # Young's modulus (Pa)
mass = 0.3             # Mass (kg)
gravity = 9.81         # Gravity (m/s^2)
theta0 = np.pi / 4     # Target bend angle (rad)
tol = 1e-5             # Adaptive RK tolerance
n_points = 200         # Sampling resolution

# ---- Geometry spline: constant + sinusoidal half-width ----
s_profile = np.linspace(0, L, 20)
w_profile = w0 + amp * np.sin(6 * np.pi * s_profile / L)
hspline = make_interp_spline(s_profile, w_profile, bc_type="natural")

# ---- Evaluation grid (used for visualization only) ----
grid = mp.matrix(np.linspace(0, L, n_points))

# ---- Run bend_samples using adaptive RKF89 ----
Fw = mass * gravity
S, F, Es = bend_samples(
    grid=grid,
    hspline=hspline,
    E=E,
    Fsin=mp.mpf(Fw),
    Fcos=True,                # Use sideforce shooting mode
    theta0=theta0,
    tol=mp.mpf(tol),
    T=T
)

# ---- Extract results ----
theta = [-1 * float(f[1]) for f in F]
Ms = np.array([float(f[0]) for f in F])
ws = np.array([float(hspline(float(s))) for s in S])
Is = (1 / 12) * (2 * ws) ** 3 * T
energy_density = Ms**2 / Is
normalized_energy = energy_density / np.max(energy_density)

# ---- Integrate flexure shape ----
x, z = integrate_xz(theta, S)
x = -x
z = -z

# ---- Plot profile + flexure shape with colormap ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Flexure profile (half-width)
axs[0].plot(s_profile, w_profile, color='magenta')
axs[0].set_title("Flexure profile (Half-width)")
axs[0].set_xlabel("s (m)")
axs[0].set_ylabel("w(s) (m)")
axs[0].grid(True)

# Flexure shape with energy colormap
points = np.array([z, x]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = Normalize(vmin=0, vmax=np.max(normalized_energy))
lc = LineCollection(segments, cmap='cool', norm=norm)
lc.set_array(normalized_energy)
lc.set_linewidth(2)
line = axs[1].add_collection(lc)

axs[1].set_xlim([0, L])
axs[1].set_ylim([min(x) - 0.01 * L, max(x) + 0.01 * L])
axs[1].set_aspect('equal')
axs[1].set_title("Flexure Shape (colored by relative energy)")
axs[1].set_xlabel("z (m)")
axs[1].set_ylabel("x (m)")

# Add colorbar
cbar = fig.colorbar(line, ax=axs[1], pad=0.02)
cbar.set_label("Relative energy density")

plt.tight_layout()
plt.show()
