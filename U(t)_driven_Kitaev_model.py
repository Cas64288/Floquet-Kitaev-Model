# Code written to compute the total Unitary evolution operator for the periodically driven Kitaev honeycomb model in 2D 
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, a0):
    dx_k = (Jy * np.sin(kx * a0) + Jz * np.sin(ky * a0))
    dy_k = (Jx + Jy * np.cos(kx * a0) + Jz * np.cos(ky * a0))
    sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    H_k =  (dx_k * sig_x) + (dy_k * sig_y)
    return H_k

# print(kitaev_hamiltonian(13, 3, 1, 1, 1, 1.0))

a0 = 1.0
T = 2*np.pi
t_step = T / 6 
J0 =  0.9 / 2
kx = np.linspace(0, 2* np.pi, 100, endpoint=False)  # 0 and 2pi are the same point in the BZ, so we can exclude 2pi to avoid duplication
ky = np.linspace(0, 2* np.pi, 100, endpoint=False)  # Doesnt matter that much but 0 = 2pi here 

# 6 pulses per period 
pulse = [
    (J0, 0, J0),    # T/6 Jx and Jz 
    (J0, 0, 0),    # 2*T/6 Jx 
    (J0, J0, 0),    # 3*T/6 Jx and Jy
    (0, J0, 0),    # 4*T/6 Jy 
    (0, J0, J0),    # 5*T/6 Jy and Jz 
    (0, 0, J0),    #  T Jz 
]
eigvals = np.zeros((len(kx), len(ky), 2), dtype=complex)
U_total_k = np.zeros((len(kx), len(ky), 2, 2), dtype=complex) # used later 
unitary_ops = []
unitary_ops_neg = []
for i, kx_val in enumerate(kx):
    for j, ky_val in enumerate(ky):
        U_total = np.eye(2, dtype=complex)
        for (Jx, Jy, Jz) in pulse:
            U = expm(-1j * kitaev_hamiltonian(kx_val, ky_val, Jx, Jy, Jz, a0) * t_step)
            U_total = U @ U_total
            U_total_k[i, j] = U_total
        eigvals[i, j, :] = np.linalg.eigvals(U_total)
epsilon_T = -np.angle(eigvals)  # as eigenvalues are exp(-i*epsilon*T)


# Full 2D plot of bands
for i in range(len(ky)):
    epsilon_ky_vals = epsilon_T[:, i, :]
    bands_sorted1 = np.sort(epsilon_ky_vals, axis=1)
    plt.plot(kx, bands_sorted1[:, 0], color='limegreen')
    plt.plot(kx, bands_sorted1[:, 1], color='limegreen')
plt.title('Floquet Quasi-energy Bulk Bands for Driven Kitaev Model (N=100)')
plt.xlabel(r'$k_x a_0$')    
plt.ylabel(r'$\varepsilon T$')
plt.xlim([0, 2*np.pi])
plt.ylim([-np.pi, np.pi])
plt.savefig('test_n_100.png')
plt.show()


# Test if hamiltonian has particle-hole symmetry
tau_y = np.array([[0, -1j],
                  [1j,  0]], dtype=complex)
U_p_2 = tau_y   # P = U_p_2 K

def check_phs_U_total(U_total_k):
    """
    U_p U(k)^* U_p^† = U(k) for all k.
    """
    for i in range(U_total_k.shape[0]):
        for j in range(U_total_k.shape[1]):
            U = U_total_k[i, j]
            U_phs = U_p_2 @ U.conj() @ U_p_2.conj().T
            diff = U_phs - U
            err = np.linalg.norm(diff, 'fro')
    return err
err_U = check_phs_U_total(U_total_k)
print("PHS error for U(T) =", err_U)