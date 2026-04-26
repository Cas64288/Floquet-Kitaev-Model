import numpy as np
from scipy.linalg import expm, eig
import matplotlib.pyplot as plt

# ---------- Model parameters and Floquet operator ----------

a0 = 1.0
T = 2*np.pi
t_step = T / 6
J0 = 0.9 / 2

Nkx = 50   # coarse grid is enough to test; increase for convergence
Nky = 50
kx_vals = np.linspace(0, 2*np.pi, Nkx, endpoint=False)
ky_vals = np.linspace(0, 2*np.pi, Nky, endpoint=False)

pulses = [
    (J0, 0,  J0),
    (J0, 0,  0 ),
    (J0, J0, 0 ),
    (0,  J0, 0 ),
    (0,  J0, J0),
    (0,  0,  J0),
]

def kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, a0):
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    dx_k = Jy*np.sin(kx*a0) + Jz*np.sin(ky*a0)
    dy_k = Jx + Jy*np.cos(kx*a0) + Jz*np.cos(ky*a0)
    return dx_k*sig_x + dy_k*sig_y

def floquet_unitary(kx, ky):
    U_tot = np.eye(2, dtype=complex)
    for (Jx, Jy, Jz) in pulses:
        H = kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, a0)
        U_p = expm(-1j * H * t_step)
        U_tot = U_p @ U_tot
    return U_tot

# ---------- Compute eigenvectors on the k-grid ----------

# eigvecs[n_band, ix, iy, :] is eigenvector of band n_band
eigvecs = np.zeros((2, Nkx, Nky, 2), dtype=complex)

for ix, kx in enumerate(kx_vals):
    for iy, ky in enumerate(ky_vals):
        U = floquet_unitary(kx, ky)
        vals, vecs = eig(U)
        # sort by quasienergy in (-pi,pi]
        eps = -np.angle(vals)
        order = np.argsort(eps)
        for n in range(2):
            v = vecs[:, order[n]]
            # fix arbitrary phase by normalising
            eigvecs[n, ix, iy, :] = v / np.linalg.norm(v)

# ---------- Fukui–Hatsugai–Suzuki Chern number ----------

def chern_number(eigvec_band):
    """
    eigvec_band[ix,iy,:] is eigenvector (normalized) on k-grid.
    Returns integer Chern number (float close to integer).
    """
    Nkx, Nky, dim = eigvec_band.shape
    F = 0.0 + 0.0j

    for ix in range(Nkx):
        ix1 = (ix + 1) % Nkx
        for iy in range(Nky):
            iy1 = (iy + 1) % Nky

            u = eigvec_band[ix,  iy,  :]
            ux = eigvec_band[ix1, iy,  :]
            uy = eigvec_band[ix,  iy1, :]
            uxy = eigvec_band[ix1, iy1, :]

            # U(1) link variables
            Ux  = np.vdot(u,  ux) / np.abs(np.vdot(u,  ux))
            Uy  = np.vdot(ux, uxy) / np.abs(np.vdot(ux, uxy))
            Ux_ = np.vdot(uy, uxy) / np.abs(np.vdot(uy, uxy))
            Uy_ = np.vdot(u,  uy) / np.abs(np.vdot(u,  uy))

            F += np.log(Ux * Uy * np.conj(Ux_) * np.conj(Uy_))

    C = F.imag / (2*np.pi)
    return C.real

# Chern number for lower and upper Floquet bands
C_lower = chern_number(eigvecs[0])
C_upper = chern_number(eigvecs[1])

print("Chern number (lower band) ≈", C_lower)
print("Chern number (upper band) ≈", C_upper)
