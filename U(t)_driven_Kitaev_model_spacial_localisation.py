# Edge localization in the periodically driven Kitaev honeycomb model
# Test: reproduce the Floquet quasi-energy spectrum with OBC in y
# and visualise edge-localised Floquet eigenstates.

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def kitaev_ft_Hamiltonian_FT_OBC(kx, Jx, Jy, Jz, a0, N, threshold=1e-10):
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    ky_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    bigH = np.zeros((2*N, 2*N), dtype=complex)
    for y in range(N):
        for y1 in range(N):
            H_ft = np.zeros((2, 2), dtype=complex)
            for ky in ky_vals:
                dx_k = Jy * np.sin(kx * a0) + Jz * np.sin(ky * a0)
                dy_k = Jx + Jy * np.cos(kx * a0) + Jz * np.cos(ky * a0)
                H_k = dx_k * sig_x + dy_k * sig_y
                H_ft += np.exp(-1j * ky * (y - y1)) * H_k
            H_ft /= N
            bigH[2*y:2*y+2, 2*y1:2*y1+2] = H_ft 
    # Open BC in y: cut the coupling between first and last rows
    bigH[0:2, -2:] = 0
    bigH[-2:, 0:2] = 0
    return bigH


N = 20
kx_vals = np.linspace(0, 2*np.pi, 20)
J0 = 0.45
a0 = 1.0
T = 2 * np.pi
t_step = T / 6
pulse = [
    (J0, 0, J0),   # T/6: Jx and Jz
    (J0, 0, 0),    # 2T/6: Jx
    (J0, J0, 0),   # 3T/6: Jx and Jy
    (0, J0, 0),    # 4T/6: Jy
    (0, J0, J0),   # 5T/6: Jy and Jz
    (0, 0, J0),    # 6T/6: Jz
]

eigvals = np.zeros((len(kx_vals), 2*N), dtype=complex)
eigvecs = np.zeros((len(kx_vals), 2*N, 2*N), dtype=complex)

for i, kx_val in enumerate(kx_vals):
    U_total = np.eye(2*N, dtype=complex)
    for (Jx, Jy, Jz) in pulse:
        H = kitaev_ft_Hamiltonian_FT_OBC(kx_val, Jx, Jy, Jz, a0, N)
        U = expm(-1j * H * t_step)
        U_total = U @ U_total
    vals, vecs = np.linalg.eig(U_total)
    eigvals[i, :] = vals
    eigvecs[i, :, :] = vecs

# Floquet quasi-energies εT in (-π, π]
epsilon_T = -np.angle(eigvals)

# Spatial probability density for each eigenstate
spatial_prob = np.zeros((len(kx_vals), 2*N, N), dtype=float)
for i in range(len(kx_vals)):
    for n in range(2*N):
        eign_func = eigvecs[i, :, n]
        for y in range(N):
            prob_A = np.abs(eign_func[2*y])**2
            prob_B = np.abs(eign_func[2*y + 1])**2
            spatial_prob[i, n, y] = prob_A + prob_B

# Edge localisation 
edge_localization = np.zeros((len(kx_vals), 2*N), dtype=float)
for i in range(len(kx_vals)):
    for n in range(2*N):
        edge_prob = spatial_prob[i, n, 0] + spatial_prob[i, n, -1]
        total_prob = np.sum(spatial_prob[i, n, :])
        edge_localization[i, n] = edge_prob / (total_prob + 1e-10) # avoid division by zero

# after computing epsilon_T (shape: Nk × 2N) and edge_localization L

vk = np.gradient(epsilon_T, kx_vals, axis=0)   # group velocity d(εT)/dkx
sign_v = np.sign(vk)                           # +1 right-moving, -1 left-moving

# Flatten for scatter
kx_rep   = np.repeat(kx_vals, 2*N)
eps_flat = epsilon_T.reshape(-1)
L_flat   = edge_localization.reshape(-1)
sign_flat = sign_v.reshape(-1)

# Choose marker by chirality
markers = np.where(sign_flat >= 0, '>', '<')   # right: '>', left: '<'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: same as before (bands as lines)
bands_sorted = np.sort(epsilon_T, axis=1)
for band in range(bands_sorted.shape[1]):
    ax1.plot(kx_vals, bands_sorted[:, band],
             color='limegreen', lw=0.5, alpha=0.6)
ax1.set_title('Floquet Quasi-energy Bands (OBC in y, N=20)')
ax1.set_xlabel(r'$k_x a_0$')
ax1.set_ylabel(r'$\varepsilon T$')
ax1.set_xlim([0, 2*np.pi])
ax1.set_ylim([-np.pi, np.pi])
ax1.grid(alpha=0.3)

# Right panel: scatter coloured by edge localisation and shaped by direction
cmap = plt.cm.inferno_r

for m in ['>', '<']:
    mask = (markers == m)
    ax2.scatter(kx_rep[mask], eps_flat[mask],
                c=L_flat[mask],
                cmap=cmap, vmin=0.0, vmax=1.0,
                s=6, alpha=0.9, marker=m)

# Optional: horizontal guide lines bracketing, say, the upper band
eps_upper_min = 1.0   # choose by eye from spectrum
eps_upper_max = 2.0
ax2.axhline(eps_upper_min, color='white', ls='--', lw=0.6)
ax2.axhline(eps_upper_max, color='white', ls='--', lw=0.6)

ax2.set_xlabel(r'$k_x a_0$')
ax2.set_ylabel(r'$\varepsilon T$')
ax2.set_title('Edge localisation and chirality in Floquet spectrum')
ax2.set_xlim([0, 2*np.pi])
ax2.set_ylim([-np.pi, np.pi])

cbar = plt.colorbar(ax2.collections[0], ax=ax2,
                    label='Edge localisation probability')
plt.tight_layout()
plt.show()


# Yippe it works!
