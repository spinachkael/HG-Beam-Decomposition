from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass, rotate
from scipy.ndimage import zoom
from PIL import Image

# CONFIGURATION

DIRECT_OPTIMISATION = True
# Input coefficients in magnitude/phase form
mode_magnitudes = [0.7, 0.8, 1, 1, 0.1, 0.7]
mode_phases = [1, -0.2, 1, 1.1, 0.9, 0.9]
mode_list = [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2)]

wavelength = 633e-9

# Grid parameters 
N = 1024 #resolution
w0 = 0.5e-3 #beam waist 
L = 8.0 * w0 #window size

# File paths
gaussian_near_path = r"KaelData20260122\Pure Gaussian.bmp"
gaussian_far_path = r"KaelData20260122\Pure Gaussiam FF.bmp"
test_near_path = r"KaelData20260122\0.7exp(1j) 0.8exp(-0.2j) 1exp(1j) 1exp(1.1j) 0.1exp(0.9j) 0.7exp(0.9j)  n.bmp"
test_far_path = r"KaelData20260122\0.7exp(1j) 0.8exp(-0.2j) 1exp(1j) 1exp(1.1j) 0.1exp(0.9j) 0.7exp(0.9j)   FF.bmp"

# Algorithm parameters
rotation_angle = -186.0
relaxation_beta = 0.7
gs_iterations = 70
propagation_distances = [0, 0.5, 1, 2, 3, 5]

# IMAGE PROCESSING

def load_image(filepath):
    img = Image.open(filepath).convert('L')
    img_array = np.array(img, dtype=np.float32)
    if img_array.max() > 0:
        img_array = img_array / img_array.max()
    return img_array

def find_beam_center(image):
    threshold = image.max() * 0.1
    mask = image > threshold
    if mask.sum() > 0:
        return center_of_mass(image * mask)
    return center_of_mass(image)

def align_image(reference_image, target_image, rotation_angle=0.0):
    #rotates and centers 
    # Rotate images (experiment axis is off)
    if rotation_angle != 0:
        target_image = rotate(target_image, -rotation_angle, reshape=False, mode='constant')
        reference_image_active = rotate(reference_image, -rotation_angle, reshape=False, mode='constant')
    else:
        reference_image_active = reference_image
    
    # Find the optical axis using the reference image
    ref_center_y, ref_center_x = find_beam_center(reference_image_active)
    
    #  Determine the final canvas size (max dimension of either image)
    h_ref, w_ref = reference_image_active.shape
    h_target, w_target = target_image.shape
    final_size = max(h_ref, w_ref, h_target, w_target)
    
    ref_square = np.zeros((final_size, final_size), dtype=reference_image_active.dtype)
    target_square = np.zeros((final_size, final_size), dtype=target_image.dtype)
    
    # Calculate offsets to place the Reference Beam Center at the Canvas Center
    start_y = final_size // 2 - int(ref_center_y)
    start_x = final_size // 2 - int(ref_center_x)
    
    # Define helper to handle array slicing 
    def paste_safe(src, dst, sy, sx):
        h, w = src.shape
        # Calculate destination bounds (clipping to canvas size)
        dy1, dy2 = max(0, sy), min(final_size, sy + h)
        dx1, dx2 = max(0, sx), min(final_size, sx + w)
        
        # Calculate source bounds (handling negative start coordinates)
        sy1, sy2 = max(0, -sy), min(h, h - (sy + h - final_size))
        sx1, sx2 = max(0, -sx), min(w, w - (sx + w - final_size))
        
        dst[dy1:dy2, dx1:dx2] = src[sy1:sy2, sx1:sx2]
        
    # Apply alignment
    paste_safe(reference_image_active, ref_square, start_y, start_x)
    paste_safe(target_image, target_square, start_y, start_x)
    
    return ref_square, target_square, final_size


# GEOMETRIC CALIBRATION HELPER FUNCTIONS

def calibrate_geometric_scale(gaussian_near, gaussian_far):
    #Calculates zoom factor needed between near and far field using reference gaussians
    
    print("Calibrating geometric scaling (FFT vs Lens)...")
    
    # 1. Compute the theoretical FFT of the near field
    theoretical_far = fftshift(fft2(ifftshift(gaussian_near)))
    theoretical_far_intensity = np.abs(theoretical_far)**2
    
    # 2. Measure widths 
    def get_width(img):
        # Threshold to remove background noise which inflates the second moment
        # Any pixel below 5% of max brightness is treated as darkness/noise
        threshold = np.max(img) * 0.05 
        img_thresh = np.copy(img)
        img_thresh[img_thresh < threshold] = 0
        # Normalize
        total_intensity = img_thresh.sum()
        if total_intensity == 0:
            return 0.0
            
        img_thresh = img_thresh / total_intensity
        
        y, x = np.indices(img.shape)
        # Recalculate center of mass on the CLEAN image
        cy, cx = center_of_mass(img_thresh)
        
        r2 = (x - cx)**2 + (y - cy)**2
        width = np.sqrt(np.sum(img_thresh * r2))
        return width

    width_fft = get_width(theoretical_far_intensity)
    width_cam = get_width(gaussian_far)
    
    #  Calculate Scale
    zoom_factor = width_fft / width_cam
    print(f"     Geometric Scale Factor:  {zoom_factor:.4f}")
    
    return zoom_factor

def apply_geometric_correction(img, zoom_factor):
    zoomed = zoom(img, zoom_factor, order=1)
    return zoomed

def process_experimental_images():
    # Load Raw Images
    gaussian_near_raw = load_image(gaussian_near_path)
    gaussian_far_raw = load_image(gaussian_far_path)
    test_near_raw = load_image(test_near_path)
    test_far_raw = load_image(test_far_path)
    
    # Pre-align Gaussian pair to calculate Scale Factor
    temp_near, _, _ = align_image(gaussian_near_raw, gaussian_near_raw, rotation_angle)
    temp_far, _, _ = align_image(gaussian_far_raw, gaussian_far_raw, rotation_angle)
    
    # Calculate the missing physics factor
    scale_factor = calibrate_geometric_scale(temp_near, temp_far)
    
    # Apply Geometric Correction to raw Far Field images
    print(f"Applying scale factor {scale_factor:.4f} to far field images...")
    gaussian_far_raw = apply_geometric_correction(gaussian_far_raw, scale_factor)
    test_far_raw = apply_geometric_correction(test_far_raw, scale_factor)
    
    # Rotate Far Field by 180 degrees to match FFT orientation
    gaussian_far_raw = rotate(gaussian_far_raw, 180, reshape=False)
    test_far_raw = rotate(test_far_raw, 180, reshape=False)
    
    # Standard Alignment & Processing
    gaussian_near_aligned, _, gaussian_size = align_image(gaussian_near_raw, gaussian_near_raw, rotation_angle)
    gaussian_far_aligned, _, _ = align_image(gaussian_far_raw, gaussian_far_raw, rotation_angle)
    
    _, test_near_aligned, _ = align_image(gaussian_near_raw, test_near_raw, rotation_angle)
    _, test_far_aligned, _ = align_image(gaussian_far_raw, test_far_raw, rotation_angle)
    
    final_size = max(
        gaussian_near_aligned.shape[0], 
        gaussian_far_aligned.shape[0],
        test_near_aligned.shape[0],
        test_far_aligned.shape[0]
    )
    
    def resize_to_final(img, final_size):
        if img.shape[0] == final_size:
            return img
        if img.shape[0] > final_size:
            start = (img.shape[0] - final_size) // 2
            return img[start:start+final_size, start:start+final_size]
        
        pad = (final_size - img.shape[0]) // 2
        resized = np.zeros((final_size, final_size), dtype=img.dtype)
        resized[pad:pad+img.shape[0], pad:pad+img.shape[1]] = img
        return resized
    
    gaussian_near_aligned = resize_to_final(gaussian_near_aligned, final_size)
    gaussian_far_aligned = resize_to_final(gaussian_far_aligned, final_size)
    test_near_aligned = resize_to_final(test_near_aligned, final_size)
    test_far_aligned = resize_to_final(test_far_aligned, final_size)
    
    # Clean up negatives/noise
    gaussian_near_aligned = np.maximum(gaussian_near_aligned, 0)
    gaussian_far_aligned = np.maximum(gaussian_far_aligned, 0)
    test_near_aligned = np.maximum(test_near_aligned, 0)
    test_far_aligned = np.maximum(test_far_aligned, 0)
    
    print(f"Processing complete. Final image size: {final_size}×{final_size}")
    
    return gaussian_near_aligned, gaussian_far_aligned, test_near_aligned, test_far_aligned, final_size

# CALIBRATION

def calibrate_far_field_scaling(gaussian_near, gaussian_far):
    #Calbrate intensity scaling factor for far field for physics consistency
    
    # Normalize inputs to max 1 
    g_near_norm = gaussian_near / np.max(gaussian_near)
    g_far_norm = gaussian_far / np.max(gaussian_far)
    
    #  Get the raw FFT energy
    raw_fft = fftshift(fft2(ifftshift(g_near_norm)))
    raw_fft_intensity = np.abs(raw_fft)**2
    
    # Calculate energies
    energy_fft = np.sum(raw_fft_intensity)
    energy_camera = np.sum(g_far_norm)
    
    # Find Scale
    # We want: energy_fft * (scale^2) = energy_camera
    # So: scale = sqrt(energy_camera / energy_fft)
    scale_factor = np.sqrt(energy_camera / energy_fft)
    
    print(f"Optimal scaling factor (Energy Balance): {scale_factor:.6e}")
    return scale_factor

# BASIS GENERATION

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

def field_from_coeffs(coeffs, basis_modes):
    field = np.zeros_like(basis_modes[0], dtype=complex)
    for coeff, mode in zip(coeffs, basis_modes):
        field += coeff * mode
    return field

def coeffs_from_field(field, basis_modes, dx, dy):
    coeffs = []
    for mode in basis_modes:
        inner = np.sum(field * mode.conj()) * dx * dy
        coeffs.append(inner)
    return np.array(coeffs)

# PHASE RETRIEVAL ALGORITHMS

def create_gaussian_mask(shape, sigma_r=0.85, order=8):
    #gaussian mask to put over beam image to reduce noise effects
    rows, cols = shape
    y = np.linspace(-1, 1, rows)
    x = np.linspace(-1, 1, cols)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    return np.exp(-((R / sigma_r)**2)**order)

def align_global_phase(coeffs_retrieved, coeffs_reference=None):
    #aligns all phases to dominant mode (all phases given as a differnce from this)
    if coeffs_reference is not None:
        idx_max = np.argmax(np.abs(coeffs_reference))
        phase_ref = np.angle(coeffs_reference[idx_max])
        phase_ret = np.angle(coeffs_retrieved[idx_max])
        phase_diff = phase_ref - phase_ret
    else:
        idx_max = np.argmax(np.abs(coeffs_retrieved))
        phase_ret = np.angle(coeffs_retrieved[idx_max])
        phase_diff = -phase_ret
        
    return coeffs_retrieved * np.exp(1j * phase_diff)

def gs_phase_retrieval_calibrated(I_near, I_far, basis_modes, dx, dy, far_field_scale, max_iter=100, beta=0.7):
    E_near = np.sum(I_near)
    E_far = np.sum(I_far)
    I_far_balanced = I_far * (E_near / E_far)
    
    mask = create_gaussian_mask(I_near.shape, sigma_r=0.85, order=8)
    phase0 = (np.random.rand(*I_near.shape) - 0.5) * 0.1
    psi = np.sqrt(np.maximum(I_near, 1e-10)) * np.exp(1j * phase0)
    
    for it in range(max_iter):
        psi_prev = psi.copy()
        psi = psi * mask
        psi_far = fftshift(fft2(ifftshift(psi)))
        
        current_phase_far = np.angle(psi_far)
        psi_far_constrained = np.sqrt(np.maximum(I_far_balanced, 1e-10)) * np.exp(1j * current_phase_far)
        psi_prime = ifftshift(ifft2(fftshift(psi_far_constrained)))
        
        psi_prime = psi_prime * mask
        current_phase_near = np.angle(psi_prime)
        psi_new = np.sqrt(np.maximum(I_near, 1e-10)) * np.exp(1j * current_phase_near)
        
        if it > 5 and beta > 0:
            combined_phase = np.angle(beta * psi_new + (1 - beta) * psi_prev)
            psi = np.sqrt(np.maximum(I_near, 1e-10)) * np.exp(1j * combined_phase)
        else:
            psi = psi_new
            
    coeffs = coeffs_from_field(psi, basis_modes, dx, dy)
    return psi, coeffs

def optimize_coeffs_calibrated(I_near, I_far, basis_modes, dx, dy, init_coeffs, far_field_scale):
    num_modes = len(basis_modes)
    
    def loss(coeffs_flat):
        coeffs = coeffs_flat[:num_modes] + 1j * coeffs_flat[num_modes:]
        psi0 = field_from_coeffs(coeffs, basis_modes)
        psi_far = fftshift(fft2(ifftshift(psi0))) * far_field_scale
        near_error = np.sum((np.abs(psi0)**2 - I_near)**2)
        far_error = np.sum((np.abs(psi_far)**2 - I_far)**2)
        return near_error + far_error
    
    x0 = np.concatenate([init_coeffs.real, init_coeffs.imag])
    res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
    
    coeffs_opt = res.x[:num_modes] + 1j * res.x[num_modes:]
    norm = np.sqrt(np.sum(np.abs(coeffs_opt)**2))
    coeffs_opt = coeffs_opt / norm if norm > 1e-10 else coeffs_opt
    
    psi_opt = field_from_coeffs(coeffs_opt, basis_modes)
    return psi_opt, coeffs_opt

def phase_retrieval_calibrated(I_near, I_far, basis_modes, dx, dy, far_field_scale):
    psi, coeffs = gs_phase_retrieval_calibrated(
        I_near, I_far, basis_modes, dx, dy, far_field_scale,
        max_iter=gs_iterations, beta=relaxation_beta
    )
    
    if DIRECT_OPTIMISATION:
        psi, coeffs = optimize_coeffs_calibrated(
        I_near, I_far, basis_modes, dx, dy, coeffs, far_field_scale
    )
    
    return psi, coeffs



def propagate_field(psi, z, L, wavelength=633e-9, return_L=False):
    #propogates beam to different z values as set by array in config
    
    if z == 0:
        if return_L: return psi, L
        return psi

    MAX_N = 2048  #max resolution set as propogation can cause crashes at higher z
    
    # determine window size (L)
    w0_est = 0.5e-3
    zR = (np.pi * w0_est**2) / wavelength
    expansion_factor = np.sqrt(1 + (z / zR)**2)
    
    # Standard Gaussian beam window
    L_physics = 8.0 * w0_est * expansion_factor
    
    # Geometric expansion
    # maintain the aspect ratio of the window growth relative to the beam
    L_required = max(L, L_physics)
    
    # determine pixel size
    N_current = psi.shape[0]
    dx_current = L / N_current
    
    # constraint for ensuring memory isnt exceeded
    dx_min_memory = L_required / MAX_N
    
    # at larger z values, ripple artifacts are seen
    #this constraint for dx is to reduce this
    dx_min_physics = (wavelength * z) / L_required
    
    # Select the limiting (largest) pixel size
    dx_new = max(dx_current, dx_min_memory, dx_min_physics)
    
    # resample input
    # must downsample the INPUT field to the new resolution
    
    # Calculate how many pixels the *original* window L occupies in the *new* grid
    N_input_new = int(L / dx_new)
    
    # Ensure reasonable minimum size to avoid destroying signal
    N_input_new = max(32, N_input_new)
    
    if N_input_new < N_current:
        # Downsample input 
        scale = N_input_new / N_current
        psi_real = zoom(psi.real, scale, order=1)
        psi_imag = zoom(psi.imag, scale, order=1)
        psi_resampled = psi_real + 1j * psi_imag
    else:
        psi_resampled = psi
        N_input_new = psi_resampled.shape[0]
        
    # Now pad this small input window to the full L_required
    N_target = int(L_required / dx_new)
    
    # Ensure even for FFT
    if N_target % 2 != 0: N_target += 1
    
    pad_total = N_target - N_input_new
    if pad_total > 0:
        pad_width = pad_total // 2
        # Apply soft window to edges of input to kill boundary artifacts
        h, w = psi_resampled.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        # Mix window: only apply it near edges
        win = win**0.1 
        psi_padded = np.pad(psi_resampled, pad_width, mode='constant')
    else:
        psi_padded = psi_resampled
        
    # Update actual params
    N_prop = psi_padded.shape[0]
    L_prop = N_prop * dx_new
    
    #ASM propogation
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N_prop, dx_new)
    fy = np.fft.fftfreq(N_prop, dx_new)
    FX, FY = np.meshgrid(fx, fy)
    
    # Transfer Function
    H_phase = z * 2 * np.pi * np.sqrt(np.maximum(0, (1/wavelength)**2 - FX**2 - FY**2))
    H = np.exp(1j * H_phase)
    
    # Bandlimit
    H[(FX**2 + FY**2) > (1/wavelength)**2] = 0
    
    psi_prop = ifft2(fft2(psi_padded) * H)
    
    if return_L:
        return psi_prop, L_prop
    return psi_prop

# VISUALIZATION

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
    
    # care about pixels where the true beam is
    # or where the retrieved beam Iincorrectly is
    mask = (m_norm > threshold) | (r_norm > threshold)
    
    # Safety check: if image is empty (e.g. at start), return global RMSE
    if np.sum(mask) == 0:
        return np.sqrt(np.mean((m_norm - r_norm)**2))
    
    # Calculate RMSE only on the masked pixels
    diff = m_norm[mask] - r_norm[mask]
    
    mse = np.mean(diff**2)
    return np.sqrt(mse)

def plot_results(test_near_norm, psi_comparison, psi_retrieved, coeffs_true, coeffs_retrieved, L_actual, test_far_norm, far_field_scale):
    # Plot 1: Near field comparison
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate the true far field for comparison
    psi_true_far = fftshift(fft2(ifftshift(psi_comparison))) * far_field_scale
    psi_retrieved_far = fftshift(fft2(ifftshift(psi_retrieved))) * far_field_scale
    
    near_images = [
        test_near_norm / np.max(test_near_norm),
        np.abs(psi_comparison)**2 / np.max(np.abs(psi_comparison)**2),
        np.abs(psi_retrieved)**2 / np.max(np.abs(psi_retrieved)**2)
    ]
    near_titles = ['Processed Near Field', 'True Near Field', 'Retrieved Near Field']
    
    vmin_near = min([img.min() for img in near_images])
    vmax_near = max([img.max() for img in near_images])
    
    for ax, title, img in zip(axes1, near_titles, near_images):
        im = ax.imshow(img, cmap='hot', extent=[-L_actual, L_actual, -L_actual, L_actual],
                      vmin=vmin_near, vmax=vmax_near)
        ax.set_title(title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Plot 2: Far field comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    far_images = [
        test_far_norm / np.max(test_far_norm),
        np.abs(psi_true_far)**2 / np.max(np.abs(psi_true_far)**2),
        np.abs(psi_retrieved_far)**2 / np.max(np.abs(psi_retrieved_far)**2)
    ]
    far_titles = ['Processed Far Field', 'True Far Field', 'Retrieved Far Field']
    
    vmin_far = min([img.min() for img in far_images])
    vmax_far = max([img.max() for img in far_images])
    
    for ax, title, img in zip(axes2, far_titles, far_images):
        im = ax.imshow(img, cmap='hot', extent=[-L_actual, L_actual, -L_actual, L_actual],
                      vmin=vmin_far, vmax=vmax_far)
        ax.set_title(title)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Plot 3: Coefficient comparison
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mode_labels = [f'({m},{n})' for m, n in mode_list]
    x_pos = np.arange(len(mode_labels))
    
    input_mags = np.abs(coeffs_true)
    retrieved_mags = np.abs(coeffs_retrieved)
    
    ax1.bar(x_pos - 0.2, input_mags, 0.4, label='True', alpha=0.7)
    ax1.bar(x_pos + 0.2, retrieved_mags, 0.4, label='Retrieved', alpha=0.7)
    ax1.set_xlabel('Mode')
    ax1.set_title('Magnitude Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mode_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    idx_dom_true = np.argmax(np.abs(coeffs_true))
    idx_dom_ret = np.argmax(np.abs(coeffs_retrieved))
    rel_phases_true = np.abs(np.angle(coeffs_true) - np.angle(coeffs_true[idx_dom_true]))
    rel_phases_ret = np.abs(np.angle(coeffs_retrieved) - np.angle(coeffs_retrieved[idx_dom_ret]))
    rel_phases_true = np.mod(rel_phases_true + np.pi, 2*np.pi) - np.pi
    rel_phases_ret = np.mod(rel_phases_ret + np.pi, 2*np.pi) - np.pi
    
    ax2.bar(x_pos - 0.2, rel_phases_true, 0.4, label='True', alpha=0.7)
    ax2.bar(x_pos + 0.2, rel_phases_ret, 0.4, label='Retrieved', alpha=0.7)
    ax2.set_xlabel('Mode')
    ax2.set_title('Relative Phase Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mode_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # PLOT 4: Propagation comparison
    fig4, axes4 = plt.subplots(2, len(propagation_distances), figsize=(20, 8))
    
    for i, z in enumerate(propagation_distances):
        # Propagate with return_full=True to get the uncropped view
        psi_true_prop, L_prop = propagate_field(psi_comparison, z, L_actual, wavelength, return_L=True)
        psi_ret_prop = propagate_field(psi_retrieved, z, L_actual, wavelength)
        
        if np.all(psi_true_prop == 0) and z > 0:
            axes4[0, i].text(0, 0, "HALTED\n(High Load)", ha='center', color='red')
            axes4[1, i].text(0, 0, "HALTED", ha='center', color='red')
            continue
        
        #  Calculate Intensity
        I_true = np.abs(psi_true_prop)**2
        I_ret = np.abs(psi_ret_prop)**2
        
        # Calculate RMSE (On the expanded grid to capture tails)
        rmse_val = calculate_intensity_rmse(I_true, I_ret)
        
        #  Normalize for display
        true_prop_norm = I_true / np.max(I_true)
        ret_prop_norm = I_ret / np.max(I_ret)
        
        # Define Dynamic Extent (Zoom out as L_prop increases)
        # L_prop is the full physical width of the grid
        extent_val = [-L_prop, L_prop, -L_prop, L_prop]
        
        # Top row: True field
        im1 = axes4[0, i].imshow(true_prop_norm, cmap='hot', extent=extent_val)
        axes4[0, i].set_title(f'True: z = {z} m\nFOV: {2*L_prop*1000:.1f} mm') # Show Field of View size
        axes4[0, i].set_xlabel('x (m)')
        
        # Bottom row: Retrieved field
        im2 = axes4[1, i].imshow(ret_prop_norm, cmap='hot', extent=extent_val)
        axes4[1, i].set_title(f'Retrieved: z = {z} m\nRMSE: {rmse_val:.4f}')
        axes4[1, i].set_xlabel('x (m)')

        # Remove y labels for inner plots to clean up
        if i > 0:
            axes4[0, i].set_yticks([])
            axes4[1, i].set_yticks([])
        else:
            axes4[0, i].set_ylabel('y (m)')
            axes4[1, i].set_ylabel('y (m)')
            
    plt.tight_layout()
    
    return fig1, fig2, fig3, fig4


def main_coherent():
    print("starting Processing...")
    
    #initialize Comparison Data
    #Convert input coefficients for the "True" comparison field
    comparison_coeffs = np.array([mag * np.exp(1j * phase) for mag, phase in zip(mode_magnitudes, mode_phases)])
    comparison_norm = np.sqrt(np.sum(np.abs(comparison_coeffs)**2))
    comparison_coeffs_norm = comparison_coeffs / comparison_norm if comparison_norm > 0 else comparison_coeffs
    
    #Process & Align Experimental Images ---
    # Load and geometrically align images
    gaussian_near, gaussian_far, test_near, test_far, final_size = process_experimental_images()
    
    #Calibrate Grid Size (L) from Gaussian Image ---
    def measure_waist_radius_pixels(img):
        threshold = np.max(img) * 0.05 
        img_thresh = np.copy(img)
        img_thresh[img_thresh < threshold] = 0
        img_thresh = img_thresh / (img_thresh.sum() + 1e-10)
        
        y, x = np.indices(img.shape)
        cy, cx = center_of_mass(img_thresh)
        r2 = (x - cx)**2 + (y - cy)**2
        # w = sqrt(2 * <r^2>) for a Gaussian beam
        second_moment = np.sum(img_thresh * r2)
        return np.sqrt(2 * second_moment)

    w0_pixels_measured = measure_waist_radius_pixels(gaussian_near)
    
    # Back-calculate L to force the Simulation Grid to match the Camera Grid pixels
    N_actual = final_size
    L_actual = (w0 * N_actual) / (2 * w0_pixels_measured)
    
    #Prepare Simulation Grid
    x = np.linspace(-L_actual, L_actual, N_actual)
    y = np.linspace(-L_actual, L_actual, N_actual)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Normalize and Filter Experimental Data ---
    test_near_norm = np.maximum(test_near / np.max(test_near), 1e-10)
    test_far_norm = np.maximum(test_far / np.max(test_far), 1e-10)
    
    # Filter Far-Field Noise
    # Create a soft-edged mask to suppress the 'circle of noise' around the far field
    ff_mask = create_gaussian_mask(test_far_norm.shape, sigma_r=0.85, order=4)
    test_far_norm = test_far_norm * ff_mask
    
    # Far Field Calibration & Basis Generation
    far_field_scale = calibrate_far_field_scaling(gaussian_near, gaussian_far)
    basis_modes = generate_orthonormal_basis(X, Y, w0, dx, dy)
    
    # Generate true field for visual comparison
    psi_comparison = field_from_coeffs(comparison_coeffs_norm, basis_modes)
    
    #Run Phase Retrieval
    print("Running phase retrieval algorithm...")
    psi_retrieved_raw, coeffs_raw = phase_retrieval_calibrated(
        test_near_norm, test_far_norm, basis_modes, dx, dy, far_field_scale
    )
    
    # algorithm cant distinguis conjugate phases
    # Compare retrieved coefficients vs their complex conjugates
    coeffs_flipped = np.conj(coeffs_raw)
    
    # Generate fields for both possibilities
    psi_std = field_from_coeffs(coeffs_raw, basis_modes)
    psi_flip = field_from_coeffs(coeffs_flipped, basis_modes)
    
    # Calculate RMSE to see which version matches the experimental near field
    rmse_std = calculate_intensity_rmse(test_near_norm, np.abs(psi_std)**2)
    rmse_flip = calculate_intensity_rmse(test_near_norm, np.abs(psi_flip)**2)
    
    if rmse_flip < rmse_std:
        coeffs_best = coeffs_flipped
    else:
        coeffs_best = coeffs_raw

    # Final Alignment and Reporting
    # Align global phase to the comparison coefficients for plotting purposes
    coeffs_retrieved = align_global_phase(coeffs_best, comparison_coeffs_norm)
    psi_retrieved = field_from_coeffs(coeffs_retrieved, basis_modes)
    
    # Plotting
    print("Generating output plots...")
    fig1, fig2, fig3, fig4 = plot_results(test_near_norm, psi_comparison, psi_retrieved, 
                                          comparison_coeffs_norm, coeffs_retrieved, 
                                          L_actual, test_far_norm, far_field_scale)
    
    # Save results
    fig1.savefig('near_field_comparison.png', dpi=150, bbox_inches='tight')
    fig2.savefig('far_field_comparison.png', dpi=150, bbox_inches='tight')
    fig3.savefig('coefficient_comparison.png', dpi=150, bbox_inches='tight')
    fig4.savefig('propagation_comparison.png', dpi=150, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main_coherent()