# Code to implement the time vortex defect within the driven Kitaev model
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Fourier transformed Kitaev Hamiltonian in both kx and ky.
# Full Hamiltonian is is real space, FT k_x ->x OBC then FT k_y -> y and OBC

def kitaev_ft_Hamiltonian_FT(Jx, Jy, Jz, a0, N_x, N_y):
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    kx_vals = np.linspace(0, 2*np.pi, N_x, endpoint=False)
    ky_vals = np.linspace(0, 2*np.pi, N_y, endpoint=False)

    # kx → x for each ky
    H_x_ky = []
    for ky in ky_vals:
        H_x = np.zeros((2*N_x, 2*N_x), dtype=complex)
        for x in range(N_x):
            for x1 in range(N_x):
                H_ft_x = np.zeros((2, 2), dtype=complex)
                for kx in kx_vals:
                    dx_k = Jy * np.sin(kx * a0) + Jz * np.sin(ky * a0)
                    dy_k = Jx + Jy * np.cos(kx * a0) + Jz * np.cos(ky * a0)
                    H_k = dx_k * sig_x + dy_k * sig_y
                    H_ft_x += np.exp(-1j * kx * (x - x1)) * H_k
                H_x[2*x:2*x+2, 2*x1:2*x1+2] = H_ft_x / N_x
        H_x[0:2, -2:] = H_x[-2:, 0:2] = 0  # OBC in x
        H_x_ky.append(H_x)

    # ky → y using the list H_x_ky
    final_H = np.zeros((2*N_x*N_y, 2*N_x*N_y), dtype=complex)
    for y in range(N_y):
        for y1 in range(N_y):
            H_ft_y = np.zeros((2*N_x, 2*N_x), dtype=complex)
            for j_ky, ky in enumerate(ky_vals):
                H_ft_y += np.exp(-1j * ky * (y - y1)) * H_x_ky[j_ky]
            y_loc = 2 * N_x * y
            y1_loc = 2 * N_x * y1
            final_H[y_loc:y_loc+2*N_x, y1_loc:y1_loc+2*N_x] = H_ft_y / N_y

    final_H[0:2*N_x, -2*N_x:] = final_H[-2*N_x:, 0:2*N_x] = 0  # OBC in y
    return final_H

def site_pos(i, Nx):
    # Maps matrix index to unit cell coords, with each cell having 2 sites
    cell = i // 2
    y = cell // Nx
    x = cell % Nx
    return x, y

def theta_func(i, j, Nx, Ny):
    xi, yi = site_pos(i, Nx)
    xj, yj = site_pos(j, Nx)
    # location of vortex core
    x0 = (Nx - 1)/2
    y0 = (Ny - 1)/2
    # bond midpoint relative to vortex core at (x0, y0)
    x_mid = 0.5 * (xi + xj) - x0
    y_mid = 0.5 * (yi + yj) - y0
    return np.arctan2(y_mid, x_mid)


def pulse_protocall_perturb(J0, a0, N_x, N_y, delta):
    # Creates 6 unitary pulses for the time vortex protocol
    # delta is used for further analysis
    pulses = [
        (J0 + delta, 0,          J0 + delta),
        (J0 + delta, 0,          0),
        (J0 + delta, J0 + delta, 0),
        (0,          J0 + delta, 0),
        (0,          J0 + delta, J0 + delta),
        (0,          0,          J0 + delta),
    ]
    H_list = []
    for (Jx, Jy, Jz) in pulses:
        H_list.append(kitaev_ft_Hamiltonian_FT(Jx, Jy, Jz, a0, N_x, N_y))
    return H_list

def pulse_index(t, T):
    # Maps time t to pulse
    dt = T / 6.0
    t_mod = t % T
    n = int(t_mod // dt)
    if n == 6:  # just in case of rounding
        n = 5
    return n

def floquet_vortex(T, H_pulses, N_x, N_y, Nt=60):
    dt = T / Nt
    dim = H_pulses[0].shape[0]
    U = np.eye(dim, dtype=complex)

    entries = []
    for i in range(dim):
        for j in range(dim):
            theta_ij = theta_func(i, j, N_x, N_y)
            entries.append((i, j, theta_ij))

    for m in range(Nt):
        t = m * dt
        H_t = np.zeros((dim, dim), dtype=complex)
        for (i, j, theta_ij) in entries:
            t_loc = (t + theta_ij * T / (2.0 * np.pi)) % T
            n = pulse_index(t_loc, T)
            H_t[i, j] = H_pulses[n][i, j]
        U = expm(-1j * H_t * dt) @ U

    return U

def floquet_no_vortex(T, H_pulses, Nt=60):
    dt = T / Nt
    dim = H_pulses[0].shape[0]
    U = np.eye(dim, dtype=complex)
    for m in range(Nt):
        t = m * dt
        n = pulse_index(t, T)
        H_t = H_pulses[n]
        U = expm(-1j * H_t * dt) @ U
    return U

# params
N_x = 15
N_y = 15
J0 = 0.45
a0 = 1.0
T = 2*np.pi
Nt = 100  # time steps for Floquet evolution

delta_drive = 0  # current perturbation strength, used to see when particle hole symmetry breaks 

H_pulses = pulse_protocall_perturb(J0, a0, N_x, N_y, delta=delta_drive)
U_vortex = floquet_vortex(T, H_pulses, N_x, N_y, Nt=Nt)

U_no_vortex = floquet_no_vortex(T, H_pulses, Nt=Nt)

evals_v, evecs_v = np.linalg.eig(U_vortex)
quasi_v = -np.angle(evals_v)

evals_nv, evecs_nv = np.linalg.eig(U_no_vortex)
quasi_nv = -np.angle(evals_nv)

# single quasienergy windows around π and 0
delta_pi = 1e-1   # window around π
delta_0  = 1e-1   # window around 0

near_pi_v = []
near_pi_nv = []
near_0_v = []
near_0_nv = []

for m, eps in enumerate(quasi_v):
    # window around π
    dist_pi = min(abs(eps - np.pi), abs(eps + np.pi))
    if dist_pi < delta_pi:
        near_pi_v.append(m)
    # window around 0
    if abs(eps) < delta_0:
        near_0_v.append(m)

for m, eps in enumerate(quasi_nv):
    dist_pi = min(abs(eps - np.pi), abs(eps + np.pi))
    if dist_pi < delta_pi:
        near_pi_nv.append(m)
    if abs(eps) < delta_0:
        near_0_nv.append(m)

dim = evecs_v.shape[0]

# LDOS with vortex (π)
ldos_pi_v = np.zeros(dim, dtype=float)
for m in near_pi_v:
    psi = evecs_v[:, m]
    ldos_pi_v += np.abs(psi)**2
if np.max(ldos_pi_v) > 0:
    ldos_pi_v /= np.max(ldos_pi_v)

ldos_grid_pi_v = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_pi_v[y, x] += ldos_pi_v[i]

# LDOS without vortex (π)
ldos_pi_nv = np.zeros(dim, dtype=float)
for m in near_pi_nv:
    psi = evecs_nv[:, m]
    ldos_pi_nv += np.abs(psi)**2
if np.max(ldos_pi_nv) > 0:
    ldos_pi_nv /= np.max(ldos_pi_nv)

ldos_grid_pi_nv = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_pi_nv[y, x] += ldos_pi_nv[i]

if np.max(ldos_grid_pi_nv) > 0:
    ldos_grid_pi_nv = ldos_grid_pi_nv / np.max(ldos_grid_pi_nv)
if np.max(ldos_grid_pi_v) > 0:
    ldos_grid_pi_v = ldos_grid_pi_v / np.max(ldos_grid_pi_v)

# LDOS with vortex (0)
ldos_0_v = np.zeros(dim, dtype=float)
for m in near_0_v:
    psi = evecs_v[:, m]
    ldos_0_v += np.abs(psi)**2
if np.max(ldos_0_v) > 0:
    ldos_0_v /= np.max(ldos_0_v)

ldos_grid_0_v = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_0_v[y, x] += ldos_0_v[i]

# LDOS without vortex (0)
ldos_0_nv = np.zeros(dim, dtype=float)
for m in near_0_nv:
    psi = evecs_nv[:, m]
    ldos_0_nv += np.abs(psi)**2
if np.max(ldos_0_nv) > 0:
    ldos_0_nv /= np.max(ldos_0_nv)

ldos_grid_0_nv = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_0_nv[y, x] += ldos_0_nv[i]

if np.max(ldos_grid_0_nv) > 0:
    ldos_grid_0_nv = ldos_grid_0_nv / np.max(ldos_grid_0_nv)
if np.max(ldos_grid_0_v) > 0:
    ldos_grid_0_v = ldos_grid_0_v / np.max(ldos_grid_0_v)


# per state LODS 

i_core = np.argmax(ldos_pi_v)  # site with max π-LDOS

eps = quasi_v  # all quasienergies
core_weight = np.abs(evecs_v[i_core, :])**2

# normalise just for colouring
if core_weight.max() > 0:
    core_weight = core_weight / core_weight.max()

# Builds site 

def honeycomb_coords(x, y, sub, a0=1.0):
    a1 = np.array([1.0, 0.0]) * a0
    a2 = np.array([0.5, np.sqrt(3)/2]) * a0
    dA = np.array([0.0, 0.0]) * a0
    dB = np.array([0.0, 1.0/np.sqrt(3)]) * a0
    R = x * a1 + y * a2
    return R + (dA if sub == 0 else dB)

def build_site_coordinates(Nx, Ny, a0=1.0):
    dim = 2 * Nx * Ny
    xs = np.zeros(dim)
    ys = np.zeros(dim)
    subs = np.zeros(dim, dtype=int)
    for i in range(dim):
        x_cell, y_cell = site_pos(i, Nx)
        sub = i % 2
        r = honeycomb_coords(x_cell, y_cell, sub, a0=a0)
        xs[i], ys[i] = r[0], r[1]
        subs[i] = sub
    return xs, ys, subs

def draw_honeycomb_bonds(Nx, Ny, a0=1.0, lw=0.4, alpha=0.35, tol=0.15):
    A = []
    B = []
    for x in range(Nx):
        for y in range(Ny):
            A.append(honeycomb_coords(x, y, 0, a0=a0))
            B.append(honeycomb_coords(x, y, 1, a0=a0))
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    ell0 = np.linalg.norm(honeycomb_coords(0, 0, 0, a0=a0) -
                          honeycomb_coords(0, 0, 1, a0=a0))
    ell_min = (1.0 - tol) * ell0
    ell_max = (1.0 + tol) * ell0
    for rA in A:
        d = np.linalg.norm(B - rA, axis=1)
        nn = np.where((d >= ell_min) & (d <= ell_max))[0]
        for k in nn:
            rB = B[k]
            plt.plot([rA[0], rB[0]], [rA[1], rB[1]],
                     color="k", lw=lw, alpha=alpha)

xs, ys, subs = build_site_coordinates(N_x, N_y, a0=a0)

# Quasienergy spectrum with time votex 

sort_idx = np.argsort(eps)
eps_sorted = eps[sort_idx]
c_sorted = core_weight[sort_idx]

plt.figure(figsize=(8, 5))
plt.title('Quasienergy Spectrum with Time Vortex')
plt.xlabel('State index (sorted by ε)')
plt.ylabel('Quasienergy')

sc = plt.scatter(
    np.arange(len(eps_sorted)),
    eps_sorted,
    c=c_sorted,
    cmap='plasma_r',
    s=15,
    marker='o'
)
plt.axhline(np.pi, color='r', linestyle='--', label=r'$\pi$')
plt.legend(loc='lower right')

cbar = plt.colorbar(sc)
cbar.set_label(r'Local DOS near $\pi$ (|ε−π|<δ)')

plt.tight_layout()
plt.show()

# Threshold the vortex LDOS (Isolates core from bulk)
#### For testing
#cut = 0.01 # keep only 80% + so we can distinctly see the bulk and vortex core 
ldos_core = ldos_pi_v.copy()
#ldos_core[ldos_core < cut] = 0.0
#ldos_core = np.clip(ldos_core, 0.0, None)

# Flatten grid LDOS for both cases (here still using π grids)
ldos_flat_nv = np.zeros(dim)
ldos_flat_v = np.zeros(dim)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_flat_nv[i] = ldos_grid_pi_nv[y, x]
    ldos_flat_v[i] = ldos_grid_pi_v[y, x]

ldos_flat_nv = np.clip(ldos_flat_nv, 0.0, None)
ldos_flat_v = np.clip(ldos_flat_v, 0.0, None)

# So scales match 
vmin = 0.0
vmax = max(ldos_flat_nv.max(), ldos_flat_v.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# left plot no vortex
ax = axes[0]
ax.set_title(r'LDOS near $\pi$ (No Vortex)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')

plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc_nv = ax.scatter(xs, ys, c=ldos_flat_nv, s=25,
                   cmap='plasma_r', vmin=vmin, vmax=vmax, marker='o')

# Right plot, time vortex 
ax = axes[1]
ax.set_title(r'LDOS near $\pi$ (Time Vortex)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')
plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc_v = ax.scatter(xs, ys, c=ldos_core, s=25,
                  cmap='plasma_r', vmin=vmin, vmax=vmax, marker='o')

# circle around core location 
x0, y0 = 10.5, 6  # <- put your vortex coords here
core_circle = Circle((x0, y0), radius=0.4,
                     edgecolor='blue', facecolor='none', lw=2)
ax.add_patch(core_circle)

cbar_v = fig.colorbar(sc_v, ax=ax, fraction=0.05, pad=0.04)
cbar_v.set_label(r'Local DOS near $\pi$')

vortex_handle = Line2D([0], [0], marker='o', ms=10, mfc='none',
                       mec='blue', linestyle='None',
                       label='Vortex Core')

ax.legend(handles=[vortex_handle], loc='upper right')
plt.savefig("ldos_no_vortex_vs_vortex_threshold.png", dpi=300)
plt.show()

ldos0_flat_nv = np.zeros(dim)
ldos0_flat_v  = np.zeros(dim)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos0_flat_nv[i] = ldos_grid_0_nv[y, x]
    ldos0_flat_v[i]  = ldos_grid_0_v[y, x]

ldos0_flat_nv = np.clip(ldos0_flat_nv, 0.0, None)
ldos0_flat_v  = np.clip(ldos0_flat_v, 0.0, None)

# Common colour range for 0-mode LDOS
vmin0 = 0.0
vmax0 = max(ldos0_flat_nv.max(), ldos0_flat_v.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# No time vortex near LDOS 0 
ax = axes[0]
ax.set_title(r'LDOS near $0$ (No Vortex)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')

plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc0_nv = ax.scatter(xs, ys, c=ldos0_flat_nv, s=25,
                    cmap='plasma_r', vmin=vmin0, vmax=vmax0, marker='o')

# Time vortex near LDOS 0 
ax = axes[1]
ax.set_title(r'LDOS near $0$ (Time Vortex)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')
plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc0_v = ax.scatter(xs, ys, c=ldos0_flat_v, s=25,
                   cmap='plasma_r', vmin=vmin0, vmax=vmax0, marker='o')

core_circle = Circle((x0, y0), radius=0.4,
                     edgecolor='blue', facecolor='none', lw=2)
ax.add_patch(core_circle)

cbar0 = fig.colorbar(sc0_v, ax=axes.ravel().tolist(), fraction=0.05, pad=0.04)
cbar0.set_label(r'Local DOS near $0$')
vortex_handle = Line2D([0], [0], marker='o', ms=10, mfc='none',
                       mec='blue', linestyle='None',
                       label='Vortex Core')

ax.legend(handles=[vortex_handle], loc='upper right')

plt.savefig("ldos_0_no_vortex_vs_vortex.png", dpi=300)
plt.show()
