# Multi-vortex implementation in the Kitaev honeycomb model  with
# periodic boundary conditions 
# 

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Original Bloch Hamiltonian in full k space 
def kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, a0):
    dx_k = Jy * np.sin(kx * a0) + Jz * np.sin(ky * a0)
    dy_k = Jx + Jy * np.cos(kx * a0) + Jz * np.cos(ky * a0)

    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    return dx_k * sig_x + dy_k * sig_y

# Original unitary in K space 
def floquet_unitary(kx, ky, T, pulse_params, a0):
    U = np.eye(2, dtype=complex)
    for Jx, Jy, Jz, dt in pulse_params:
        H_k = kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, a0)
        U = expm(-1j * H_k * dt) @ U
    return U

# Lattice site builing 
def site_pos(i, Nx):
    # Maps site i to unit cell coords 
    cell = i // 2
    y = cell // Nx
    x = cell % Nx
    return x, y


def site_pos_with_sub(i, Nx, Ny):
    # Maps site index i to (x,y, sublattice)
    cell = i // 2
    sub = i % 2
    y = cell // Nx
    x = cell % Nx
    return x, y, sub

]#builds real space kitav lattice 
def honeycomb_coords(x, y, sub, a0=1.0):
    a1 = np.array([1.0, 0.0]) * a0
    a2 = np.array([0.5, np.sqrt(3) / 2]) * a0
    dA = np.array([0.0, 0.0])
    dB = np.array([0.0, 1.0 / np.sqrt(3) * a0])
    R = x * a1 + y * a2
    return R + (dA if sub == 0 else dB)


def cell_index(x, y, Nx, Ny):
    return (y % Ny) * Nx + (x % Nx)


def site_index(x, y, sub, Nx, Ny):
    return 2 * cell_index(x, y, Nx, Ny) + sub

#generates nearest neighbour bonds to connet sites on lattice 
# We have for each entry (Site a, Site b, bond_type, bond_midpoint)
def bond_list_pbc(Nx, Ny, a0=1.0):
    bonds = []
    for x in range(Nx):
        for y in range(Ny):
            iA = site_index(x, y, 0, Nx, Ny)
            rA = honeycomb_coords(x, y, 0, a0=a0)

            iBx = site_index(x, y, 1, Nx, Ny)
            rBx = honeycomb_coords(x, y, 1, a0=a0)
            bonds.append((iA, iBx, 'x', 0.5 * (rA + rBx)))

            iBy = site_index(x - 1, y, 1, Nx, Ny)
            rBy = honeycomb_coords(x - 1, y, 1, a0=a0)
            bonds.append((iA, iBy, 'y', 0.5 * (rA + rBy)))

            iBz = site_index(x, y - 1, 1, Nx, Ny)
            rBz = honeycomb_coords(x, y - 1, 1, a0=a0)
            bonds.append((iA, iBz, 'z', 0.5 * (rA + rBz)))
    return bonds



# Bravis lattice vectors for a hexagonal lattice 
# With required displacment
def torus_metric_vectors(Nx, Ny, a0=1.0):
    L1 = Nx * np.array([1.0, 0.0]) * a0
    L2 = Ny * np.array([0.5, np.sqrt(3) / 2]) * a0
    M = np.column_stack([L1, L2])
    Minv = np.linalg.inv(M)
    return M, Minv

def minimum_image_displacement(r, r0, M, Minv):
    coeffs = Minv @ (r - r0)
    coeffs -= np.round(coeffs)
    return M @ coeffs

# Computes the total vortex field at r 
# We say +1 for vortex and -1 for antivortex 
def phase_from_vortices(r, vortex_list, M, Minv):
    z = 1.0 + 0.0j
    for (x0, y0, q) in vortex_list:
        r0 = honeycomb_coords(x0, y0, 0, a0=1.0)
        dr = minimum_image_displacement(r, r0, M, Minv)
        w = dr[0] + 1j * dr[1]
        # Avoid division by zero exactly at a vortex core
        if np.abs(w) < 1e-12:
            continue
        if q == 1:
            z *= w / np.abs(w)
        elif q == -1:
            z *= np.conj(w) / np.abs(w)
        else:
            z *= (w / np.abs(w)) ** q
    return np.angle(z)

# Assigns phase to each bond midpoint causing the temporal effects 
def bond_phase(i, j, Nx, Ny, vortex_list, a0=1.0):
    xi, yi, subi = site_pos_with_sub(i, Nx, Ny)
    xj, yj, subj = site_pos_with_sub(j, Nx, Ny)
    ri = honeycomb_coords(xi, yi, subi, a0=a0)
    rj = honeycomb_coords(xj, yj, subj, a0=a0)
    rmid = 0.5 * (ri + rj)
    M, Minv = torus_metric_vectors(Nx, Ny, a0=a0)
    return phase_from_vortices(rmid, vortex_list, M, Minv)

# Real space implemtation of the lattice 
def kitaev_hamiltonian_realspace_from_bonds(Jx, Jy, Jz, a0, Nx, Ny):
    dim = 2 * Nx * Ny
    H = np.zeros((dim, dim), dtype=complex)
    Jdict = {'x': Jx, 'y': Jy, 'z': Jz}

    for iA, iB, bond_type, _ in bond_list_pbc(Nx, Ny, a0=a0):
        J = Jdict[bond_type]
        if J != 0.0:
            H[iA, iB] += 1j * J
            H[iB, iA] += -1j * J

    return H
# Pulse protocol
# Same as before, real space hamiltonian needed here
def pulse_protocol_realspace(J0, a0, Nx, Ny):
    pulses = [
        (J0, 0,  J0),
        (J0, 0,  0),
        (J0, J0, 0),
        (0,  J0, 0),
        (0,  J0, J0),
        (0,  0,  J0),
    ]
    return [
        kitaev_hamiltonian_realspace_from_bonds(Jx, Jy, Jz, a0, Nx, Ny)
        for (Jx, Jy, Jz) in pulses
    ]

# To identify pulse index 
def pulse_index(t, T):
    dt = T / 6.0
    t_mod = t % T
    n = int(t_mod // dt)
    if n == 6:
        n = 5
    return n

# Floquet unitary for no vortex 
def floquet_no_vortex(T, H_pulses, Nt=60):
    dt = T / Nt
    dim = H_pulses[0].shape[0]
    U = np.eye(dim, dtype=complex)

    for m in range(Nt):
        t = m * dt
        n = pulse_index(t, T)
        U = expm(-1j * H_pulses[n] * dt) @ U

    return U

#  Multi-Vortex implementation 
def floquet_multi_vortex(T, H_pulses, Nx, Ny, vortex_list, Nt=60, a0=1.0):
    dt = T / Nt
    dim = H_pulses[0].shape[0]
    U = np.eye(dim, dtype=complex)

    entries = []
    for i in range(dim):
        for j in range(dim):
            theta_ij = bond_phase(i, j, Nx, Ny, vortex_list, a0=a0)
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

# Draw nearest-neighbour bonds, connected according to sublattice A and B positions

def build_site_coordinates(Nx, Ny, a0=1.0):
    dim = 2 * Nx * Ny
    xs = np.zeros(dim)
    ys = np.zeros(dim)
    subs = np.zeros(dim, dtype=int)

    for i in range(dim):
        x_cell, y_cell, sub = site_pos_with_sub(i, Nx, Ny)
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

    ell0 = np.linalg.norm(
        honeycomb_coords(0, 0, 0, a0=a0) - honeycomb_coords(0, 0, 1, a0=a0)
    )
    ell_min = (1.0 - tol) * ell0
    ell_max = (1.0 + tol) * ell0

    for rA in A:
        d = np.linalg.norm(B - rA, axis=1)
        nn = np.where((d >= ell_min) & (d <= ell_max))[0]
        for k in nn:
            rB = B[k]
            plt.plot([rA[0], rB[0]], [rA[1], rB[1]], color='k', lw=lw, alpha=alpha)


# params
N_x = 15
N_y = 15
J0 = 0.45
a0 = 1.0
T = 2 * np.pi
Nt = 100

# Moves location of the vortex, and last argument changes the vorticity 
# Currently set for a vortex anti vortex pair
vortex_list = [
    (3, 7, +1),
    (10, 2, -1),
]
H_pulses = pulse_protocol_realspace(J0, a0, N_x, N_y)
U_vortex = floquet_multi_vortex(T, H_pulses, N_x, N_y, vortex_list, Nt=Nt, a0=a0)
U_no_vortex = floquet_no_vortex(T, H_pulses, Nt=Nt)

evals_v, evecs_v = np.linalg.eig(U_vortex)
quasi_v = -np.angle(evals_v)

evals_nv, evecs_nv = np.linalg.eig(U_no_vortex)
quasi_nv = -np.angle(evals_nv)


delta_pi = 1e-1 # energy window
near_pi_v = []
near_pi_nv = []

for m, eps in enumerate(quasi_v):
    dist_pi = min(abs(eps - np.pi), abs(eps + np.pi))
    if dist_pi < delta_pi:
        near_pi_v.append(m)

for m, eps in enumerate(quasi_nv):
    dist_pi = min(abs(eps - np.pi), abs(eps + np.pi))
    if dist_pi < delta_pi:
        near_pi_nv.append(m)

dim = evecs_v.shape[0]

ldos_pi_v = np.zeros(dim, dtype=float)
for m in near_pi_v:
    psi = evecs_v[:, m]
    ldos_pi_v += np.abs(psi) ** 2
if np.max(ldos_pi_v) > 0:
    ldos_pi_v /= np.max(ldos_pi_v)

ldos_grid_pi_v = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_pi_v[y, x] += ldos_pi_v[i]

ldos_pi_nv = np.zeros(dim, dtype=float)
for m in near_pi_nv:
    psi = evecs_nv[:, m]
    ldos_pi_nv += np.abs(psi) ** 2
if np.max(ldos_pi_nv) > 0:
    ldos_pi_nv /= np.max(ldos_pi_nv)

ldos_grid_pi_nv = np.zeros((N_y, N_x), dtype=float)
for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_grid_pi_nv[y, x] += ldos_pi_nv[i]

if np.max(ldos_grid_pi_nv) > 0:
    ldos_grid_pi_nv /= np.max(ldos_grid_pi_nv)
if np.max(ldos_grid_pi_v) > 0:
    ldos_grid_pi_v /= np.max(ldos_grid_pi_v)

#print('energy', quasi_v[near_pi_v])


# Plotting
xs, ys, subs = build_site_coordinates(N_x, N_y, a0=a0)

ldos_flat_nv = np.zeros(dim)
ldos_flat_v = np.zeros(dim)

for i in range(dim):
    x, y = site_pos(i, N_x)
    ldos_flat_nv[i] = ldos_grid_pi_nv[y, x]
    ldos_flat_v[i] = ldos_grid_pi_v[y, x]

ldos_flat_nv = np.clip(ldos_flat_nv, 0.0, None)
ldos_flat_v = np.clip(ldos_flat_v, 0.0, None)

vmin = 0.0
vmax = max(ldos_flat_nv.max(), ldos_flat_v.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

ax = axes[0]
ax.set_title(r'LDOS near $\pi$ (No Vortex)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')
plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc_nv = ax.scatter(xs, ys, c=ldos_flat_nv, s=25, cmap='plasma_r',
                   vmin=vmin, vmax=vmax, marker='o')

ax = axes[1]
ax.set_title(r'LDOS near $\pi$ (Vortex-Antivortex Pair)')
ax.set_xlabel('Lx')
ax.set_ylabel('Ly')
plt.sca(ax)
draw_honeycomb_bonds(N_x, N_y, a0=a0, lw=0.4, alpha=0.25)
sc_v = ax.scatter(xs, ys, c=ldos_flat_v, s=25, cmap='plasma_r',
                  vmin=vmin, vmax=vmax, marker='o')

for (xv, yv, qv) in vortex_list:
    rv = honeycomb_coords(xv, yv, 0, a0=a0)
    edgecolor = 'blue' if qv > 0 else 'green'
    ax.add_patch(Circle((rv[0], rv[1]), radius=0.25,
                        edgecolor=edgecolor, facecolor='none', lw=2))

cbar_v = fig.colorbar(sc_v, ax=ax, fraction=0.05, pad=0.04)
cbar_v.set_label(r'Local DOS near $\pi$')

vortex_handle = Line2D([0], [0], marker='o', ms=10, mfc='none',
                       mec='blue', linestyle='None', label='Vortex')
antivortex_handle = Line2D([0], [0], marker='o', ms=10, mfc='none',
                           mec='green', linestyle='None', label='Anti-vortex')
ax.legend(handles=[vortex_handle, antivortex_handle], loc='upper right')

plt.tight_layout()
plt.show()

# it works 
