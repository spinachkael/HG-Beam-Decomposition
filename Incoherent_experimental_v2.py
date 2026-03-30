from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from scipy.fft import fft2, ifft2
from scipy.ndimage import center_of_mass, rotate, zoom
from PIL import Image
from scipy.ndimage import gaussian_filter

# CONFIGURATION

# Set to True if the beam composition is known (testing with a known beam).
# Set to False for an unknown beam
KNOWN_BEAM = True

# Known beam inputs 
component_files = [
    r"KaelData20260122\Pure Gaussian.bmp",
    r"KaelData20260122\Pure (1,0).bmp", 
    r"KaelData20260122\Pure (0,1).bmp",
    r"KaelData20260122\Pure (1,1).bmp",
    r"KaelData20260122\pure (2,0).bmp",
    r"KaelData20260122\Pure (0,2).bmp" 
]
true_coeffs = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1]

# Unknown beam input (only used if KNOWN_BEAM = False)
unknown_beam_path = r""

gaussian_ref_path = r"KaelData20260122\Pure Gaussian.bmp"

mode_list = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2)]
w0 = 0.5e-3  
wavelength = 633e-9 
rotation_angle = -186.0
propagation_distances = [0, 0.5, 1, 2, 3]
TIKHONOV_ALPHA = 0.5 

# IMAGE PROCESSING & ALIGNMENT

def load_image(filepath):
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)
    if img_array.max() > 0:
        img_array = img_array / img_array.max()
    return img_array

def find_beam_center(image): 
    threshold = image.max() * 0.1
    mask = image > threshold
    return center_of_mass(image * mask) if mask.sum() > 0 else center_of_mass(image)

def align_image(reference_image, target_image, rotation_angle=0.0):
    if rotation_angle != 0:
        target_image = rotate(target_image, -rotation_angle, reshape=False, mode='constant')
        reference_image_active = rotate(reference_image, -rotation_angle, reshape=False, mode='constant')
    else:
        reference_image_active = reference_image
    
    ref_center_y, ref_center_x = find_beam_center(reference_image_active)
    
    h_ref, w_ref = reference_image_active.shape
    h_target, w_target = target_image.shape
    final_size = max(h_ref, w_ref, h_target, w_target)
    
    ref_square = np.zeros((final_size, final_size), dtype=reference_image_active.dtype)
    target_square = np.zeros((final_size, final_size), dtype=target_image.dtype)
    
    start_y = final_size // 2 - int(ref_center_y)
    start_x = final_size // 2 - int(ref_center_x)
    
    def paste_safe(src, dst, sy, sx):
        h, w = src.shape
        dy1, dy2 = max(0, sy), min(final_size, sy + h)
        dx1, dx2 = max(0, sx), min(final_size, sx + w)
        sy1, sy2 = max(0, -sy), min(h, h - (sy + h - final_size))
        sx1, sx2 = max(0, -sx), min(w, w - (sx + w - final_size))
        dst[dy1:dy2, dx1:dx2] = src[sy1:sy2, sx1:sx2]
        
    paste_safe(reference_image_active, ref_square, start_y, start_x)
    paste_safe(target_image, target_square, start_y, start_x)
    
    return ref_square, target_square, final_size


def generate_orthonormal_basis(X, Y, w0, dx, dy):
    raw_modes = []
    for m, n in mode_list:
        Hm = hermite(m)
        Hn = hermite(n)
        prefactor = np.sqrt(2 / (np.pi * w0**2 * 2**(m+n) * factorial(m) * factorial(n)))
        x_term = Hm(np.sqrt(2) * X / w0)
        y_term = Hn(np.sqrt(2) * Y / w0)
        gaussian = np.exp(-(X**2 + Y**2) / w0**2)
        mode = x_term * y_term * gaussian * prefactor
        raw_modes.append(mode)
    
    #Gram-schmidt orthogonlisation, in case discrete grid causes erorrs
    ortho_modes = []
    for i, mode in enumerate(raw_modes):
        v = mode.copy()
        for j in range(i):
            proj = np.sum(v * ortho_modes[j].conj()) * dx * dy
            v = v - proj * ortho_modes[j]
        
        norm = np.sqrt(np.sum(np.abs(v)**2) * dx * dy)
        ortho_modes.append(v / norm if norm > 1e-10 else v * 0)
    
    return ortho_modes


def calculate_intensity_rmse(measured_intensity, retrieved_intensity, threshold_percent=1.0):
    #calculates rmse only within beam region/ region of interest
    #region of interest is deined as a pixel where the measured or... 
    # ...retreieved intensity is above the threshold percent of the max intensity
   
    # 1. Normalize both to [0, 1] for fair comparison
    m_max = np.max(measured_intensity) + 1e-10
    r_max = np.max(retrieved_intensity) + 1e-10
    
    m_norm = measured_intensity / m_max
    r_norm = retrieved_intensity / r_max
    
    # 2. Create the mask
    threshold = threshold_percent / 100.0
    mask = (m_norm > threshold) | (r_norm > threshold)
    
    # Safety check: if image is empty (e.g. at start), return global RMSE
    if np.sum(mask) == 0:
        return np.sqrt(np.mean((m_norm - r_norm)**2))
    
    diff = m_norm[mask] - r_norm[mask]
    return np.sqrt(np.mean(diff**2))


def propagate_field(psi, z, L, wavelength=633e-9, return_L=False):
    if z == 0:
        if return_L: return psi, L
        return psi

    MAX_N = 2048
    zR = (np.pi * w0**2) / wavelength
    expansion_factor = np.sqrt(1 + (z / zR)**2)
    L_physics = 8.0 * w0 * expansion_factor
    L_required = max(L, L_physics)
    
    N_current = psi.shape[0]
    dx_current = L / N_current
    dx_min_memory = L_required / MAX_N
    dx_min_physics = (wavelength * z) / L_required
    dx_new = max(dx_current, dx_min_memory, dx_min_physics)
    
    N_input_new = max(32, int(L / dx_new))
    
    if N_input_new < N_current:
        scale = N_input_new / N_current
        psi_real = zoom(psi.real, scale, order=1)
        psi_imag = zoom(psi.imag, scale, order=1)
        psi_resampled = psi_real + 1j * psi_imag
    else:
        psi_resampled = psi
        N_input_new = psi_resampled.shape[0]
        
    N_target = int(L_required / dx_new)
    if N_target % 2 != 0: N_target += 1
    
    pad_total = N_target - N_input_new
    if pad_total > 0:
        pad_width = pad_total // 2
        h, w = psi_resampled.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        win = win**0.1
        psi_padded = np.pad(psi_resampled * win, pad_width, mode='constant')
    else:
        psi_padded = psi_resampled
        
    N_prop = psi_padded.shape[0]
    fx = np.fft.fftfreq(N_prop, dx_new)
    fy = np.fft.fftfreq(N_prop, dx_new)
    FX, FY = np.meshgrid(fx, fy)
    
    H_phase = z * 2 * np.pi * np.sqrt(np.maximum(0, (1/wavelength)**2 - FX**2 - FY**2))
    H = np.exp(1j * H_phase)
    H[(FX**2 + FY**2) > (1/wavelength)**2] = 0
    
    psi_prop = ifft2(fft2(psi_padded) * H)
    
    if return_L:
        return psi_prop, N_prop * dx_new
    return psi_prop

# MAIN EXECUTION

def main_incoherent():
    print("Starting Incoherent Decomposition...")
    ref_raw = load_image(gaussian_ref_path)
    ref_aligned, _, final_size = align_image(ref_raw, ref_raw, rotation_angle)
    
    # Grid Calibration — threshold before second moment to avoid noise bias
    threshold = np.max(ref_aligned) * 0.05
    ref_thresh = np.copy(ref_aligned)
    ref_thresh[ref_thresh < threshold] = 0
    img_norm = ref_thresh / (ref_thresh.sum() + 1e-10)
    y_idx, x_idx = np.indices(ref_aligned.shape)
    cy, cx = center_of_mass(img_norm)
    w0_px = np.sqrt(2 * np.sum(img_norm * ((x_idx - cx)**2 + (y_idx - cy)**2)))
    L_actual = (w0 * final_size) / (2 * w0_px)
    
    x_vec = np.linspace(-L_actual, L_actual, final_size)
    X, Y = np.meshgrid(x_vec, x_vec)
    dx = x_vec[1] - x_vec[0]
    dy = dx

    basis_modes = generate_orthonormal_basis(X, Y, w0, dx, dy)

    if KNOWN_BEAM:
        # Build synthetic incoherent beam from weighted sum of component images
        true_coeffs_norm = np.array(true_coeffs, dtype=np.float64)
        true_coeffs_norm /= np.sum(true_coeffs_norm)

        synthetic_image = np.zeros((final_size, final_size))
        for i, fpath in enumerate(component_files):
            _, img_aligned, _ = align_image(ref_raw, load_image(fpath), rotation_angle)
            synthetic_image += true_coeffs_norm[i] * img_aligned
        synthetic_image /= np.max(synthetic_image)
        synthetic_image = gaussian_filter(synthetic_image, sigma=1.5)
        input_image = synthetic_image
    else:
        # Load the unknown beam image directly
        _, input_image, _ = align_image(ref_raw, load_image(unknown_beam_path), rotation_angle)
        input_image = np.maximum(input_image, 0)
        input_image /= np.max(input_image)
        input_image = gaussian_filter(input_image, sigma=1.5)
        true_coeffs_norm = None

    # Decompose
    basis_matrix = []
    for mode in basis_modes:
        intensity = np.abs(mode)**2
        basis_matrix.append((intensity / np.max(intensity)).flatten())
    
    A = np.array(basis_matrix).T
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    weights = Vt.T @ ((S / (S**2 + TIKHONOV_ALPHA**2)) * (U.T @ input_image.flatten()))
    weights = np.maximum(weights, 0)
    weights /= np.sum(weights)

    # Text output
    print("\nRetrieved Mode Weights:")
    print(f"  {'Mode':<10} {'Weight':>10} {'Power (%)':>10}")
    print(f"  {'-'*32}")
    for (m, n), w in zip(mode_list, weights):
        print(f"  ({m},{n}){'':<6} {w:>10.4f} {100*w:>9.2f}%")

    # Visualization
    print("\nGenerating Plots...")

    # Plot 1: Input field
    if KNOWN_BEAM:
        true_mathematical = np.zeros_like(input_image)
        for i, mode in enumerate(basis_modes):
            intensity = np.abs(mode)**2
            true_mathematical += true_coeffs_norm[i] * (intensity / np.max(intensity))

        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
        axes1[0].imshow(true_mathematical, cmap='hot', extent=[-L_actual, L_actual, -L_actual, L_actual])
        axes1[0].set_title("Mathematical True Field")
        axes1[0].set_xlabel('x (m)')
        axes1[0].set_ylabel('y (m)')
        axes1[1].imshow(input_image, cmap='hot', extent=[-L_actual, L_actual, -L_actual, L_actual])
        axes1[1].set_title(f"Experimental Field (RMSE: {calculate_intensity_rmse(true_mathematical, input_image):.4f})")
        axes1[1].set_xlabel('x (m)')
        axes1[1].set_ylabel('y (m)')
    else:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.imshow(input_image, cmap='hot', extent=[-L_actual, L_actual, -L_actual, L_actual])
        ax1.set_title("Input Beam")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
    plt.tight_layout()

    # Plot 2: Weight distribution
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(mode_list))
    if KNOWN_BEAM:
        ax2.bar(x_pos - 0.2, true_coeffs_norm, 0.4, label='True')
        ax2.bar(x_pos + 0.2, weights, 0.4, label='Retrieved')
        ax2.set_title("Incoherent Weight Distribution")
    else:
        ax2.bar(x_pos, weights, 0.4, label='Retrieved')
        ax2.set_title("Retrieved Incoherent Weight Distribution")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"({m},{n})" for m,n in mode_list])
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Weight')
    ax2.legend()
    plt.tight_layout()

    # Plot 3: Propagation
    if not propagation_distances:
        plt.show()
        return

    if KNOWN_BEAM:
        fig3, axes3 = plt.subplots(2, len(propagation_distances), figsize=(20, 8))
        if len(propagation_distances) == 1:
            axes3 = axes3.reshape(2, 1)
    else:
        fig3, axes3_row = plt.subplots(1, len(propagation_distances), figsize=(20, 4))
        if len(propagation_distances) == 1:
            axes3_row = [axes3_row]

    for i, z in enumerate(propagation_distances):
        test_psi, L_p = propagate_field(basis_modes[0], z, L_actual, wavelength, return_L=True)
        I_retrieved_z = np.zeros_like(test_psi, dtype=np.float32)
        if KNOWN_BEAM:
            I_true_z = np.zeros_like(test_psi, dtype=np.float32)
        
        for j, mode in enumerate(basis_modes):
            psi_z = propagate_field(mode, z, L_actual, wavelength)
            if psi_z.shape != test_psi.shape:
                scale = test_psi.shape[0] / psi_z.shape[0]
                psi_z = zoom(psi_z.real, scale, order=1) + 1j * zoom(psi_z.imag, scale, order=1)
            
            intensity = np.abs(psi_z)**2
            I_retrieved_z += weights[j] * intensity
            if KNOWN_BEAM:
                I_true_z += true_coeffs_norm[j] * intensity
            
        I_ret_norm = I_retrieved_z / np.max(I_retrieved_z) if np.max(I_retrieved_z) > 0 else I_retrieved_z
        ext = [-L_p, L_p, -L_p, L_p]

        if KNOWN_BEAM:
            I_true_norm = I_true_z / np.max(I_true_z) if np.max(I_true_z) > 0 else I_true_z
            rmse_val = calculate_intensity_rmse(I_true_norm, I_ret_norm)

            axes3[0, i].imshow(I_true_norm, cmap='hot', extent=ext)
            axes3[0, i].set_title(f"True z={z}m")
            axes3[0, i].set_xlabel('x (m)')
            axes3[1, i].imshow(I_ret_norm, cmap='hot', extent=ext)
            axes3[1, i].set_title(f"Retrieved RMSE: {rmse_val:.4f}")
            axes3[1, i].set_xlabel('x (m)')
            if i == 0:
                axes3[0, i].set_ylabel('y (m)')
                axes3[1, i].set_ylabel('y (m)')
            else:
                axes3[0, i].set_yticks([])
                axes3[1, i].set_yticks([])
        else:
            axes3_row[i].imshow(I_ret_norm, cmap='hot', extent=ext)
            axes3_row[i].set_title(f"Retrieved z={z}m\nFOV: {2*L_p*1000:.1f}mm")
            axes3_row[i].set_xlabel('x (m)')
            if i == 0:
                axes3_row[i].set_ylabel('y (m)')
            else:
                axes3_row[i].set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_incoherent()
