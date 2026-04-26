# Chern number calculation for the driven Kitaev honeycomb model 
import numpy as np
from scipy.linalg import expm, eig
import matplotlib.pyplot as plt

# Params 
a0 = 1.0
T = 2*np.pi
t_step = T / 6
J0 = 0.9 / 2

Nkx = 50   # Good enough to test for now can be increased for accuracy 
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
# Same functions as before 
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

# Note: eigvecs[n_band, ix, iy, :] is eigenvector of band n_band
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
            eigvecs[n, ix, iy, :] = v / np.linalg.norm(v)
# Computes Chern number 
def chern_number(eigvec_band):
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

print("chern number for lower band  ≈", chern_number(eigvecs[0]))
print("chern number for upper band  ≈",  chern_number(eigvecs[1])
