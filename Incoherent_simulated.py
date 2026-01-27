from math import factorial
import numpy as np
from scipy.special import hermite
import matplotlib.pyplot as plt

snr = 0  # signal to noise ratio
alpha = 0.2  # regularization parameter

#define mode pairs (n,m) to include in superposition
mode_pairs = [(0,0), (0,1), (1,0), (1,1), (2,0), (0,2), (2,1), (1,2)] 

#set up grid 
N = 1000               # Grid resolution
L = 6.0               # Window size 
w0 = 1.0              # Beam waist

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y, indexing='ij')

#create HG mode fucntions
def HG_mode(n, x, w0):
    Hn = hermite(n)
    prefactor = (2 / (np.pi * w0**2))**0.25 / np.sqrt(2**n * factorial(n))
    x_term = Hn(np.sqrt(2) * x / w0)
    gaussian = np.exp(-(x**2) / w0**2)
    return x_term * gaussian * prefactor

#  generate superposition 
max_n = max([n for n, m in mode_pairs]) if mode_pairs else 0
max_m = max([m for n, m in mode_pairs]) if mode_pairs else 0
num_modes = len(mode_pairs)

complex_weights = {}
for (n, m) in mode_pairs:
    amplitude = np.random.rand()   
    phase = 2 * np.pi * np.random.rand()
    complex_weights[(n, m)] = amplitude * np.exp(1j * phase)

# Normalize
total_power = sum(np.abs(c)**2 for c in complex_weights.values())
for key in complex_weights:
    complex_weights[key] /= np.sqrt(total_power)

#calculate true intensity_incoherent map
I_incoherent_true = np.zeros((N, N))
true_coeffs = np.zeros((max_n+1, max_m+1))

for (n, m), c in complex_weights.items():
    mode_x = HG_mode(n, X, w0)
    mode_y = HG_mode(m, Y, w0)
    I_incoherent_true += np.abs(c)**2 * np.abs(mode_x * mode_y)**2
    true_coeffs[n, m] = np.abs(c)**2

# Noise function
def add_experimental_noise(I_clean, snr_normalized):
    peak_signal = np.max(I_clean)
    noise_amplitude = snr_normalized * peak_signal
    noise = np.random.randn(*I_clean.shape) * noise_amplitude
    I_noisy = I_clean + noise
    I_noisy = np.maximum(I_noisy, 0)
    return I_noisy

# Add noise to the true intensity
I_noisy = add_experimental_noise(I_incoherent_true, snr)

def HG_mode(n, x, w0):
    Hn = hermite(n)
    prefactor = (2 / (np.pi * w0**2))**0.25 / np.sqrt(2**n * factorial(n))
    x_term = Hn(np.sqrt(2) * x / w0)
    gaussian = np.exp(-(x**2) / w0**2)
    return x_term * gaussian * prefactor

def recover_coeffs_regularised(I_measured, x, y, w0, max_n_recover, max_m_recover, alpha):
    Nx, Ny = len(x), len(y)
    I_flat = I_measured.flatten(order='C')
    X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')
    
    num_recover_modes = (max_n_recover + 1) * (max_m_recover + 1)
    A = np.zeros((Nx * Ny, num_recover_modes))
    
    idx = 0
    for n in range(max_n_recover + 1):  
        for m in range(max_m_recover + 1): 
            mode_x = HG_mode(n, X_grid, w0)
            mode_y = HG_mode(m, Y_grid, w0)
            mode_2d = np.abs(mode_x * mode_y)**2  
            A[:, idx] = mode_2d.flatten(order='C')
            idx += 1
    
    # SVD with Tikhonov regularization
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_tikhonov = S / (S**2 + alpha**2)  
    coeffs_flat = Vt.T @ (S_tikhonov * (U.T @ I_flat))
    
    coeffs_2d = coeffs_flat.reshape((max_n_recover + 1, max_m_recover + 1), order='C')
    coeffs_2d[coeffs_2d < 0] = 0
    coeffs_2d = coeffs_2d / np.sum(coeffs_2d) * np.sum(true_coeffs)
    
    return coeffs_2d, S

#perform estimation
max_n_recover = max_n
max_m_recover = max_m
estimated_coeffs, singular_values = recover_coeffs_regularised(I_noisy, x, y, w0, max_n_recover, max_m_recover, alpha)


# Extract modes
modes_to_plot = []
true_vals = []
estimated_vals = []

for (n, m) in mode_pairs:
    modes_to_plot.append(f"({n},{m})")
    true_vals.append(true_coeffs[n, m])
    estimated_vals.append(estimated_coeffs[n, m])

x_pos = np.arange(len(modes_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x_pos - width/2, true_vals, width, label='True', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, estimated_vals, width, label='Retrieved', color='orange', alpha=0.7)

ax.set_xlabel('Mode (n,m)')
ax.set_ylabel('Coefficient Value')
ax.set_title('True vs Retrieved Coefficients')
ax.set_xticks(x_pos)
ax.set_xticklabels(modes_to_plot, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()