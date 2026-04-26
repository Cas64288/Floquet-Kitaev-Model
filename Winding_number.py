# Code written to compute the winding number given the time evolution operator U(k,t) for the driven Kitaev model in 2D 
# Increase Nkx, Nky, Nt to get more accurate winding number
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
kx = np.linspace(0, 2* np.pi, 100, endpoint=False)
ky = np.linspace(0, 2* np.pi, 100, endpoint=False) 

# 6 pulses per period 
pulse = [
    (J0, 0, J0),    # T/6 Jx and Jz 
    (J0, 0, 0),    # 2*T/6 Jx 
    (J0, J0, 0),    # 3*T/6 Jx and Jy
    (0, J0, 0),    # 4*T/6 Jy 
    (0, J0, J0),    # 5*T/6 Jy and Jz 
    (0, 0, J0),    #  T Jz 
]

# Time resolved evolution (discrited time further for a more accurate winding number) 
Nt = 100                 
t_grid = np.linspace(0.0, T, Nt, endpoint=False)
dt = T / Nt
pulse_times = [t_step] * len(pulse)        # each pulse lasts t_step = T/6
pulse_cum = np.cumsum(pulse_times)         # end times of each pulse
pulse_start = np.concatenate(([0.0], pulse_cum[:-1]))

def H_of_t(kx_val, ky_val, t):
# Gives hamiltomnian at any time t by checking pulse index returns H(kx,ky) for that pulse
# find which pulse we are in
    t_mod = t % T
    idx = np.searchsorted(pulse_cum, t_mod) # find the index of the first pulse end time that is greater than t_mod
    Jx, Jy, Jz = pulse[idx]
    return kitaev_hamiltonian(kx_val, ky_val, Jx, Jy, Jz, a0)


# Time ordered evolution discretise time into Nt steps, 
# Apply the appropriate Hamiltonian for that time interval
U_t = np.zeros((len(kx), len(ky), Nt, 2, 2), dtype=complex)

for i, kx_val in enumerate(kx):
    for j, ky_val in enumerate(ky):
        U = np.eye(2, dtype=complex)
        U_t[i, j, 0] = U
        for n in range(1, Nt):
            t_mid = (t_grid[n-1] + t_grid[n]) / 2.0
            H_mid = H_of_t(kx_val, ky_val, t_mid)
            dU = expm(-1j * H_mid * dt)
            U = dU @ U
            U_t[i, j, n] = U


# Effective Floquet Hamiltonian H_eff = (i/T) log U_T with logarithm branch centered at quasienergy epsilon.
def Heff_k(U_T, eps, T):
    evals, evecs = np.linalg.eig(U_T)

    # raw quasienergies in (-pi/T, pi/T]
    eps_raw = -np.angle(evals) / T

    # shift into interval centered at eps
    shift = np.round((eps_raw - eps) * T / (2 * np.pi))
    eps_br = eps_raw - 2 * np.pi * shift / T
    Heff = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        v = evecs[:, a].reshape(2, 1)
        Heff += eps_br[a] * (v @ v.conj().T)
    return Heff

# Builds U_eps(k,t)
    #U_eps(k,t) = U(k, 2t)          for 0 <= t <= T/2
    #U_eps(k,t) = exp[-i Heff (2T-2t)] for T/2 <= t <= T

def build_U_eps(kx, ky, U_t, T, eps):
    Nkx, Nky, Nt = len(kx), len(ky), U_t.shape[2]
    U_eps = np.zeros_like(U_t, dtype=complex)

    for i in range(Nkx):
        for j in range(Nky):
            U_T = U_t[i, j, -1]
            Heff = Heff_k(U_T, eps, T)

            for n, t in enumerate(t_grid):
                if t <= T / 2:
                    t2 = 2.0 * t
                    m = int(np.floor(t2 / dt)) % Nt
                    U_eps[i, j, n] = U_t[i, j, m]
                else:
                    tau = 2.0 * T - 2.0 * t
                    V = expm(-1j * Heff * tau)
                    U_eps[i, j, n] = V
    return U_eps

# Computes 3D winding number W from U_eps(kx,ky,t) using discrete version of the formula in Rudner et al 2013
# Forward differece scheme estimates derivatives 
# W = (1 / 24 pi^2) ∫ d^3x ε^{μνλ} Tr[(U^{-1}∂_μU)(U^{-1}∂_νU)(U^{-1}∂_λU)] on the torus (kx, ky, t).
def winding_number(U_grid, kx, ky, T):
    Nkx, Nky, Nt = len(kx), len(ky), U_grid.shape[2]
    dkx = (kx[-1] - kx[0]) / (Nkx - 1)
    dky = (ky[-1] - ky[0]) / (Nky - 1)
    dt = T / Nt

    W = 0.0
    for i in range(Nkx):
        ip = (i + 1) % Nkx # periodic boundary in kx (also for ky)
        for j in range(Nky):
            jp = (j + 1) % Nky
            for n in range(Nt):
                np1 = (n + 1) % Nt

                U = U_grid[i, j, n]
                U_inv = np.linalg.inv(U)

                d_kx = (U_grid[ip, j, n] - U) / dkx
                d_ky = (U_grid[i, jp, n] - U) / dky
                d_t  = (U_grid[i, j, np1] - U) / dt

                A_x = U_inv @ d_kx
                A_y = U_inv @ d_ky
                A_t = U_inv @ d_t

                term = np.trace(
                    A_x @ A_y @ A_t
                    - A_x @ A_t @ A_y
                    + A_y @ A_t @ A_x
                    - A_y @ A_x @ A_t
                    + A_t @ A_x @ A_y
                    - A_t @ A_y @ A_x
                )

                W += term.real * dkx * dky * dt

    W *= 1.0 / (24.0 * np.pi**2)
    return W

# Compute W 
eps0 = 0.0
eps_pi = np.pi / T

print("Building U_eps for eps = 0 ...")
U_eps_0 = build_U_eps(kx, ky, U_t, T, eps0)

print("Building U_eps for eps = pi/T ...")
U_eps_pi = build_U_eps(kx, ky, U_t, T, eps_pi)

print("Computing winding numbers ...")
W_0  = winding_number(U_eps_0, kx, ky, T)
W_pi = winding_number(U_eps_pi, kx, ky, T)

print("Winding number around eps = 0     :", W_0)
print("Winding number around eps = pi/T :", W_pi)
