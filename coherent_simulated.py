from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize

# Changeable parameters
snr = 0  # signal to noise ratio (=noise/ peak signal) 
print("SNR used in this run:", snr)

mode_list = [(0,0), (0,1), (1,0), (1,1), (0,2), (2,0)]  # list of modes for HG basis

# Grid parameters
N = 600            # resolution
L = 8.0            # window size
w0 = 1.0           # beam waist

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Make HG functions
def make_HG(m, n, X, Y, w0):
    Hm = hermite(m)
    Hn = hermite(n)
    prefactor = np.sqrt(2 / (np.pi * w0**2 * 2**(m+n) * factorial(m) * factorial(n)))
    x_term = Hm(np.sqrt(2) * X / w0)
    y_term = Hn(np.sqrt(2) * Y / w0)
    gaussian = np.exp(-(X**2 + Y**2) / w0**2)
    return x_term * y_term * gaussian * prefactor

def generate_orthonormal_modes(mode_list, X, Y, w0, dx, dy):
    modes = []
    for (m,n) in mode_list:
        mode = make_HG(m, n, X, Y, w0)
        norm = np.sqrt(np.sum(np.abs(mode)**2) * dx * dy)
        modes.append(mode / norm)
    
    modes_ortho = []
    for i, mode in enumerate(modes):
        v = mode.copy()
        for j in range(i):
            proj = np.sum(v * modes_ortho[j].conj()) * dx * dy
            v = v - proj * modes_ortho[j]
        norm = np.sqrt(np.sum(np.abs(v)**2) * dx * dy)
        if norm > 1e-10:
            modes_ortho.append(v / norm)
        else:
            modes_ortho.append(v * 0)
    
    return modes_ortho

# Generate orthonormal modes
modes_ortho = generate_orthonormal_modes(mode_list, X, Y, w0, dx, dy)

def reconstruct_from_coeffs(coeffs, modes_ortho):
    field = np.zeros_like(modes_ortho[0], dtype=complex)
    for idx, coeff in enumerate(coeffs):
        field += coeff * modes_ortho[idx]
    return field

def compute_coeffs(field, modes_ortho, dx, dy):
    coeffs = []
    for mode in modes_ortho:
        inner_product = np.sum(field * mode.conj()) * dx * dy
        coeffs.append(inner_product)
    return np.array(coeffs)

# Create random amplitudes and phases for true field
random_amplitudes = np.random.rand(len(mode_list))
random_phases = 2 * np.pi * np.random.rand(len(mode_list))
random_weights = random_amplitudes * np.exp(1j * random_phases)
random_weights /= np.linalg.norm(random_weights)  # Normalize

# True field
psi0_true = reconstruct_from_coeffs(random_weights, modes_ortho)
I_near_clean = np.abs(psi0_true)**2
phase_true = np.angle(psi0_true)

# Create far field from the same field
psiF_true = fftshift(fft2(ifftshift(psi0_true))) * dx * dy / (2*np.pi)
I_far_clean = np.abs(psiF_true)**2

# Add experimental noise
def add_experimental_noise(I_clean, snr):
    peak_signal = np.max(I_clean)
    noise_amplitude = snr * peak_signal
    noise = np.random.randn(*I_clean.shape) * noise_amplitude
    I_noisy = I_clean + noise
    I_noisy = np.maximum(I_noisy, 0)
    return I_noisy

I_near_noisy = add_experimental_noise(I_near_clean, snr)
I_far_noisy = add_experimental_noise(I_far_clean, snr)

def gs_phase_retrieval(I_near, I_far, modes_ortho, dx, dy, max_iter=100):
    phase0 = np.random.rand(*I_near.shape) * 2*np.pi
    psi0 = np.sqrt(I_near) * np.exp(1j*phase0)
    
    for it in range(max_iter):
        psi0 = np.sqrt(I_near) * np.exp(1j*np.angle(psi0))
        
        coeffs = compute_coeffs(psi0, modes_ortho, dx, dy)
        psi0 = reconstruct_from_coeffs(coeffs, modes_ortho)
        
        psiF = fftshift(fft2(ifftshift(psi0))) * dx * dy / (2*np.pi)
        psiF = np.sqrt(I_far) * np.exp(1j*np.angle(psiF))
        
        psi0 = ifftshift(ifft2(fftshift(psiF))) * (2*np.pi) / (dx * dy)
    
    return psi0

def direct_optimization(I_near, I_far, modes_ortho, dx, dy, init_coeffs):
    num_modes = len(modes_ortho)
    
    def loss(coeffs_flat):
        coeffs = coeffs_flat[:num_modes] + 1j*coeffs_flat[num_modes:]
        psi0 = reconstruct_from_coeffs(coeffs, modes_ortho)
        psiF = fftshift(fft2(ifftshift(psi0))) * dx * dy / (2*np.pi)
        near_error = np.sum((np.abs(psi0)**2 - I_near)**2) * dx * dy
        far_error = np.sum((np.abs(psiF)**2 - I_far)**2) * dx * dy
        return near_error + far_error
    
    x0 = np.concatenate([init_coeffs.real, init_coeffs.imag])
    res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
    
    coeffs_opt = res.x[:num_modes] + 1j*res.x[num_modes:]
    coeffs_opt = coeffs_opt / np.linalg.norm(coeffs_opt)
    psi0_opt = reconstruct_from_coeffs(coeffs_opt, modes_ortho)
    
    return psi0_opt, coeffs_opt

# Run reconstruction with noisy data - exactly 1 GS and 1 optimization
print("Running hybrid phase retrieval...")
psi0_gs = gs_phase_retrieval(I_near_noisy, I_far_noisy, modes_ortho, dx, dy, max_iter=100)
coeffs_gs = compute_coeffs(psi0_gs, modes_ortho, dx, dy)
coeffs_gs = coeffs_gs / np.linalg.norm(coeffs_gs)

psi0_estimated, coeffs_hybrid = direct_optimization(I_near_noisy, I_far_noisy, modes_ortho, dx, dy, coeffs_gs)

# Prepare coefficients for comparison
true_coeffs = random_weights.copy()
retrieved_coeffs = coeffs_hybrid.copy()

# Correct for global phase ambiguity
global_phase = np.angle(np.sum(true_coeffs * np.conj(retrieved_coeffs)))
retrieved_coeffs_corrected = retrieved_coeffs * np.exp(1j * global_phase)


# Create coefficient bar chart
mode_labels = [f'({m},{n})' for m, n in mode_list]
x_pos = np.arange(len(mode_labels))

fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(10, 8))

# Real parts
real_true = true_coeffs.real
real_retrieved = retrieved_coeffs_corrected.real

ax_real.bar(x_pos - 0.2, real_true, width=0.4, alpha=0.8, label='True', color='blue')
ax_real.bar(x_pos + 0.2, real_retrieved, width=0.4, alpha=0.8, label='Retrieved', color='orange')
ax_real.set_xticks(x_pos)
ax_real.set_xticklabels(mode_labels)
ax_real.set_ylabel('Real part')
ax_real.set_title('Comparison of Real Parts of Coefficients')
ax_real.legend()
ax_real.grid(True, alpha=0.3)

# Imaginary parts
imag_true = true_coeffs.imag
imag_retrieved = retrieved_coeffs_corrected.imag

ax_imag.bar(x_pos - 0.2, imag_true, width=0.4, alpha=0.8, label='True', color='blue')
ax_imag.bar(x_pos + 0.2, imag_retrieved, width=0.4, alpha=0.8, label='Retrieved', color='orange')
ax_imag.set_xticks(x_pos)
ax_imag.set_xticklabels(mode_labels)
ax_imag.set_ylabel('Imaginary part')
ax_imag.set_title('Comparison of Imaginary Parts of Coefficients')
ax_imag.legend()
ax_imag.grid(True, alpha=0.3)

plt.suptitle(f'Coefficient Comparison', fontsize=14)
plt.tight_layout()
plt.show()