# Compute Floquet quasienergy bands for the periodically driven Kitaev honeycomb model in the anomalous phase
# Fourier transform in k_y-> y while keeping k_x as a good quantum number 
# Code also contains a pertubation test to see if energies are still prtected for small pertubations of the drive

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


# The Hamiltonian is first written in mixed momentum space then Fourier transformed over ky to produce a real-space description along y.
def kitaev_ft_Hamiltonian_FT_OBC(kx, Jx, Jy, Jz, a0, N):
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    ky_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    bigH = np.zeros((2*N, 2*N), dtype=complex)
    for y in range(N):
        for y1 in range(N):
            H_ft = np.zeros((2,2), dtype=complex)
            # Fourier transform from ky to real-space y,y1 matrix elements
            for ky in ky_vals:
                dx_k = Jy * np.sin(kx * a0) + Jz * np.sin(ky * a0)
                dy_k = Jx + Jy * np.cos(kx * a0) + Jz * np.cos(ky * a0)
                H_k = dx_k * sig_x + dy_k * sig_y
                H_ft += np.exp(-1j * ky * (y - y1)) * H_k
            H_ft /= N
            bigH[2*y:2*y+2, 2*y1:2*y1+2] = H_ft
    bigH[0:2, -2:] = 0
    bigH[-2:, 0:2] = 0
    #bigH = np.where(np.abs(bigH.real) < threshold, 0, bigH.real) + 1j * np.where(np.abs(bigH.imag) < threshold, 0, bigH.imag) # not needed
    return bigH
delta = 0 # testing 
N = 100 # discritisation number 
kx_vals = np.linspace(0, 2*np.pi, 100)
J0 = 0.45
a0 = 1.0
T = 2 * np.pi
t_step = T / 6
pulse = [
    (J0 + delta, 0, J0+delta),    # T/6 Jx and Jz 
    (J0+delta, 0, 0),    # 2*T/6 Jx 
    (J0+delta, J0+delta, 0),    # 3*T/6 Jx and Jy
    (0, J0+delta, 0),    # 4*T/6 Jy 
    (0, J0+delta, J0+delta),    # 5*T/6 Jy and Jz 
    (0, 0, J0+delta),    #  T Jz 
]

eigvals = np.zeros((len(kx_vals), 2*N), dtype=complex)

for i, kx_val in enumerate(kx_vals):
    U_total = np.eye(2*N, dtype=complex)
    for (Jx, Jy, Jz) in pulse:
        H = kitaev_ft_Hamiltonian_FT_OBC(kx_val, Jx, Jy, Jz, a0, N)
        U = expm(-1j * H * t_step)
        U_total = U @ U_total
    eigvals[i, :] = np.linalg.eigvals(U_total)

epsilon_T = -np.angle(eigvals)  

bands_sorted = np.sort(epsilon_T, axis=1)  # Sort at each kx (2*N bands for N=20)


for band in range(bands_sorted.shape[1]):
    plt.plot(kx_vals, bands_sorted[:, band], color='limegreen', lw=0.7, alpha=0.8)
plt.title('Floquet Quasi-energy Bands Driven Kitaev Model (perturbed, delta=0.66, N=100)') # Plot name can be changed depending on what you are doing
plt.xlabel(r'$k_x a_0$')
plt.ylabel(r'$\varepsilon T$')
plt.xlim([0, 2*np.pi])
plt.ylim([-np.pi, np.pi])
#plt.savefig('test0.png')
plt.show()
