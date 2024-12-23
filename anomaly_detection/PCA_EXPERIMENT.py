import json 
import torch
import numpy as np
import hyperspy.api as hs
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

energy_values = None

################################################################
########################### UTILS ##############################
################################################################

def find_peaks(spectrum, prominence=0.1, distance=1, plot=False):
    """
    Find peaks in the spectrum using scipy's find_peaks function.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    prominence (float): Minimum prominence of the peaks (relative to the maximum intensity)
    distance (int): Minimum distance between peaks
    plot (bool): If True, plot the spectrum with detected peaks
    
    Returns:
    tuple: (peak_positions, peak_properties)
    """
    prominence = prominence * np.max(spectrum)
    peaks, properties = signal.find_peaks(spectrum, prominence=prominence, distance=distance)
    
    if plot:
        plot_spectrum_with_peaks(spectrum, peaks, properties)
    
    return peaks, properties

def plot_spectrum_with_peaks(spectrum, peaks, properties):
    """
    Plot the spectrum with detected peaks.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    peaks (numpy.ndarray): Array of peak indices
    properties (dict): Properties of the peaks
    """
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum)
    plt.plot(peaks, spectrum[peaks], "x")
    
    # Highlighting the most prominent peak
    most_prominent = peaks[np.argmax(properties['prominences'])]
    plt.plot(most_prominent, spectrum[most_prominent], "o", color='red', markersize=10)
    
    plt.title("Spectrum with Detected Peaks (Red circle: Most Prominent)")
    plt.xlabel("Channel")
    plt.ylabel("Intensity")
    plt.show()

def find_main_peak(spectrum, plot=False):
    """
    Find the main peak in the spectrum based on highest prominence.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    plot (bool): If True, plot the spectrum with the main peak
    
    Returns:
    int: The index of the main peak
    """
    peaks, properties = find_peaks(spectrum, prominence=0.1, distance=10)
    if len(peaks) == 0:
        main_peak = np.argmax(spectrum)
    else:
        # Select the peak with the highest prominence
        main_peak = peaks[np.argmax(properties['prominences'])]
    
    if plot:
        plot_spectrum_with_peaks(spectrum, [main_peak], {'prominences': [properties['prominences'][np.argmax(properties['prominences'])]]})
    
    return main_peak

def inject_controlled_anomalies(image, anomaly_specs, energy_range=None):
    """
    Inject controlled anomalies into the EELS image.
    
    Args:
    image (numpy.ndarray): Input image of shape (height, width, energy_channels)
    anomaly_specs (list): List of tuples specifying anomalies. Each tuple can be either:
        - (type, count, cluster_size) for basic usage
        - (type, count, cluster_size, dict) where dict contains specific parameters
    energy_range (numpy.ndarray, optional): Energy values for oxygen deficiency simulation
    
    Returns:
    tuple: (anomalous_image, anomaly_mask)
    """
    anomalous_image = np.copy(image)
    height, width, _ = image.shape
    anomaly_mask = np.zeros((height, width), dtype=int)
    
    anomaly_funcs = {
        'peak_shift': peak_shift,
        'peak_broadening': peak_broadening,
        'intensity_fluctuation': intensity_fluctuation,
        'background_slope': background_slope,
        'noise_injection': noise_injection,
        'multiple_scattering': multiple_scattering,
        'cosmic_ray': cosmic_ray,
        'oxygen_deficiency': simulate_oxygen_deficiency  # Add new function
    }
    
    anomaly_colors = {
        'peak_shift': 1,
        'peak_broadening': 2,
        'intensity_fluctuation': 3,
        'background_slope': 4,
        'noise_injection': 5,
        'multiple_scattering': 6,
        'cosmic_ray': 7,
        'oxygen_deficiency': 8  # Add new color code
    }
    
    for spec in anomaly_specs:
        # Unpack specifications
        if len(spec) == 4:
            anomaly_type, count, cluster_size, params = spec
        else:
            anomaly_type, count, cluster_size = spec
            params = {}
        
        for _ in range(count):
            if anomaly_type == 'cosmic_ray':
                # Handle cosmic ray as before
                while True:
                    y = np.random.randint(0, height)
                    x = np.random.randint(0, width)
                    if anomaly_mask[y, x] == 0:
                        anomaly_mask[y, x] = anomaly_colors[anomaly_type]
                        anomalous_image[y, x] = anomaly_funcs[anomaly_type](
                            anomalous_image[y, x], **params)
                        break
            else:
                # For other anomaly types, including oxygen_deficiency
                while True:
                    center_y = np.random.randint(cluster_size, height - cluster_size)
                    center_x = np.random.randint(cluster_size, width - cluster_size)
                    
                    if cluster_size == 1:
                        y_start, y_end = center_y, center_y + 1
                        x_start, x_end = center_x, center_x + 1
                    else:
                        y_start, y_end = center_y - cluster_size // 2, center_y + cluster_size // 2
                        x_start, x_end = center_x - cluster_size // 2, center_x + cluster_size // 2
                    
                    if np.all(anomaly_mask[y_start:y_end, x_start:x_end] == 0):
                        anomaly_mask[y_start:y_end, x_start:x_end] = anomaly_colors[anomaly_type]
                        for y in range(y_start, y_end):
                            for x in range(x_start, x_end):
                                if anomaly_type == 'oxygen_deficiency' and energy_range is not None:
                                    anomalous_image[y, x] = anomaly_funcs[anomaly_type](
                                        anomalous_image[y, x], energy_range, **params)
                                else:
                                    anomalous_image[y, x] = anomaly_funcs[anomaly_type](
                                        anomalous_image[y, x], **params)
                        break
    
    return anomalous_image, anomaly_mask

def simulate_oxygen_deficiency(original_spec, energy_range=energy_values, deficiency_level=0.1, ok_edge_start=533):
    """
    Simulate oxygen deficiency with effects only applied to O K-edge region and beyond
    
    Parameters:
    -----------
    original_spec : array-like
        Original XAS spectrum
    energy_range : array-like
        Energy values
    deficiency_level : float
        Degree of oxygen deficiency to simulate (0-1)
    ok_edge_start : float
        Energy value where O K-edge region starts
    """
    # Create mask for O K-edge region and beyond
    ok_edge_mask = (energy_range >= ok_edge_start) & (energy_range <= ok_edge_start + 30)
    
    # Create modified spectrum
    modified_spec = original_spec.copy()
    
    # Only apply effects to O K-edge region
    edge_energies = energy_range[ok_edge_mask]
    edge_spec = modified_spec[ok_edge_mask]
    
    # Normalize the O K-edge portion
    edge_spec = (edge_spec - np.min(edge_spec)) / (np.max(edge_spec) - np.min(edge_spec))
    
    # Create energy normalization only for O K-edge region
    energy_normalized = (edge_energies - np.min(edge_energies)) / (np.max(edge_energies) - np.min(edge_energies))
    
    # Create sigmoid transitions for amplitude reduction
    sigmoid1 = 1 / (1 + np.exp(-(energy_normalized - 0.3) * 10))
    sigmoid2 = 1 / (1 + np.exp(-(energy_normalized - 0.6) * 15))
    
    # Combine sigmoids for complex amplitude reduction pattern
    amplitude_factor = 1 - (deficiency_level * (0.7 * sigmoid1 + 0.3 * sigmoid2))
    
    # Apply amplitude reduction only to edge region
    edge_spec *= amplitude_factor
    
    # Apply gaussian blur with varying sigma only to edge region
    sigma = deficiency_level * (0.5 + energy_normalized * 3.0)
    edge_spec = gaussian_filter1d(edge_spec, sigma=np.max(sigma))
    
    # Apply small shift only to edge region
    shift_amount = int(deficiency_level * 8)
    edge_spec = np.roll(edge_spec, -shift_amount)
    if shift_amount > 0:
        edge_spec[-shift_amount:] = edge_spec[-(shift_amount+1)]
    
    # Rescale edge_spec back to original range
    edge_range = np.max(original_spec[ok_edge_mask]) - np.min(original_spec[ok_edge_mask])
    edge_min = np.min(original_spec[ok_edge_mask])
    edge_spec = (edge_spec * edge_range) + edge_min
    
    # Insert modified edge region back into spectrum
    modified_spec[ok_edge_mask] = edge_spec
    
    return modified_spec

def peak_shift(spectrum, shift_value=25, window_size=20): 
    """
    Apply peak shift to the spectrum.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    shift_value (int, optional): If provided, shift will be randomly +/- this value
    window_size (int): Size of the window around the peak to apply the shift
    
    Returns:
    numpy.ndarray: The shifted spectrum
    """
    peak_pos = find_main_peak(spectrum)
    
    # Randomly choose between positive and negative shift
    shift = shift_value * (2 * np.random.randint(0, 2) - 1)  # Returns either +shift_value or -shift_value
    window = slice(max(0, peak_pos - window_size), min(len(spectrum), peak_pos + window_size + 1))
    shifted_spectrum = np.roll(spectrum[window], shift)
    spectrum[window] = shifted_spectrum
    return spectrum

def peak_broadening(spectrum, max_sigma=5): # WORKS OK
    peak_pos = find_main_peak(spectrum)
    window_size = 51  # Adjust this value based on your typical peak widths
    window = slice(max(0, peak_pos - window_size//2), min(len(spectrum), peak_pos + window_size//2 + 1))
    peak_region = spectrum[window]
    sigma = np.random.uniform(0, max_sigma)
    x = np.arange(-window_size//2, window_size//2 + 1)
    gaussian = np.exp(-x**2 / (2 * sigma**2))
    gaussian /= gaussian.sum()
    broadened_peak = fftconvolve(peak_region, gaussian, mode='same')
    broadened_peak *= np.sum(peak_region) / np.sum(broadened_peak)
    result = spectrum.copy()
    result[window] = broadened_peak
    return result

def intensity_fluctuation(spectrum, scale_range=None): # WORKS WELL 
    """
    Apply intensity fluctuation to the spectrum.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    scale_range (tuple): A tuple of (min_scale, max_scale). If None, default to (0.8, 1.2)
    
    Returns:
    numpy.ndarray: The spectrum with intensity fluctuation applied
    """
    if scale_range is None:
        scale_range = (0.8, 1.2)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return spectrum * scale

def background_slope(spectrum, max_slope=0.00013): # WORKS WELL 
    slope = np.random.uniform(-max_slope, max_slope)
    background = np.arange(len(spectrum)) * slope
    return spectrum + background

def noise_injection(spectrum, poisson_factor=0.0005, gaussian_std=0.09): # WORKS WELL 
    poisson_noise = np.random.poisson(poisson_factor * spectrum)
    gaussian_noise = np.random.normal(0, gaussian_std * np.mean(spectrum), len(spectrum))
    return spectrum + poisson_noise + gaussian_noise

def multiple_scattering(spectrum, scattering_factor=0.1): # WORKS WELL 
    convolved = np.convolve(spectrum, spectrum, mode='same')
    return spectrum + scattering_factor * convolved / np.max(convolved)

def cosmic_ray(spectrum, intensity_factor=10, width=3): # WORKS WELL 
    """
    Add a cosmic ray spike to the spectrum.
    
    Args:
    spectrum (numpy.ndarray): The input spectrum
    intensity_factor (float): Factor to determine the intensity of the cosmic ray
    width (int): Width of the cosmic ray peak
    
    Returns:
    numpy.ndarray: The spectrum with an added cosmic ray spike
    """
    pos = np.random.randint(len(spectrum))
    x = np.arange(2 * width + 1) - width
    gaussian_peak = gaussian(x, 0, width/3)
    start = max(0, pos - width)
    end = min(len(spectrum), pos + width + 1)
    peak_start = width - (pos - start)
    peak_end = width + (end - pos)
    spectrum[start:end] += intensity_factor * np.max(spectrum) * gaussian_peak[peak_start:peak_end]
    return spectrum

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Define anomaly types (this should be consistent with your inject_controlled_anomalies function)
anomaly_types = {
    'peak_shift': 1,
    'peak_broadening': 2,
    'intensity_fluctuation': 3,
    'background_slope': 4,
    'noise_injection': 5,
    'multiple_scattering': 6,
    'cosmic_ray': 7
}

def plot_anomaly_mask(anomaly_mask, eels_image):
    """
    Plot the anomaly mask as an overlay on the EELS SI image.
    
    Args:
    anomaly_mask (numpy.ndarray): 2D array with integer codes for anomaly types
    eels_image (numpy.ndarray): 3D EELS SI image (height, width, energy_channels)
    """
    # Create a summed image of the EELS SI data
    summed_image = np.sum(eels_image, axis=2)
    
    # Create colormap for anomalies
    anomaly_colors = ['none', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
    cmap = mcolors.ListedColormap(anomaly_colors)
    bounds = list(range(len(anomaly_colors) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create the overlay
    overlay = create_anomaly_overlay(anomaly_mask, anomaly_colors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the summed EELS image
    ax.imshow(summed_image, cmap='gray')
    
    # Overlay the anomaly mask
    ax.imshow(overlay, alpha=0.5)
    
    # Create a custom colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(anomaly_colors)) + 0.5)
    cbar.set_ticklabels(['No anomaly'] + list(anomaly_types.keys()))
    
    plt.title('Anomaly Overlay on EELS SI Image')
    plt.axis('off')
    plt.show()

def create_anomaly_overlay(anomaly_mask, colors):
    """
    Create an RGBA overlay for the anomaly mask.
    
    Args:
    anomaly_mask (numpy.ndarray): 2D array with integer codes for anomaly types
    colors (list): List of color names for each anomaly type
    
    Returns:
    numpy.ndarray: RGBA array representing the overlay
    """
    height, width = anomaly_mask.shape
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    
    for i, color in enumerate(colors[1:], start=1):  # Skip 'none' color
        mask = anomaly_mask == i
        rgba = mcolors.to_rgba(color)
        overlay[mask] = rgba
    
    return overlay

def adjust_energy_dimension(data, target_points=1312):
    """
    Adjust the energy dimension of the data to exactly 1312 points
    by centering around the middle of the available range
    
    Args:
        data: numpy array of shape (height, width, energy_points)
        target_points: desired number of points in energy dimension (default: 1312)
    
    Returns:
        adjusted_data: numpy array with energy dimension of exactly target_points
    """
    height, width, energy_points = data.shape
    
    # Find the middle point of the current energy range
    mid_idx = energy_points // 2
    half_target = target_points // 2
    
    # Calculate start and end indices
    start_idx = max(0, mid_idx - half_target)
    end_idx = min(energy_points, start_idx + target_points)
    
    # If we hit the upper bound, adjust the start index
    if end_idx == energy_points:
        start_idx = end_idx - target_points
    # If we hit the lower bound, adjust the end index
    elif start_idx == 0:
        end_idx = target_points
    
    # Crop the data
    adjusted_data = data[..., start_idx:end_idx]
    
    print(f"Original energy points: {energy_points}")
    print(f"Adjusted energy points: {adjusted_data.shape[-1]}")
    
    return adjusted_data

################################################################
########################### UTILS ##############################
################################################################

def split_into_shards(data, shard_size=24):
    """
    Split data into 24x24 shards with exactly 1312 energy points
    
    Args:
        data: numpy array of shape (height, width, energy_points)
        shard_size: size of each square shard
        target_energy_points: desired number of points in energy dimension
    
    Returns:
        shards: list of numpy arrays, each of shape (shard_size, shard_size, target_energy_points)
        positions: list of tuples containing (i, j) start positions of each shard
    """
    # First adjust the energy dimension
    height, width, energy_points = data.shape
        
    shards = []
    positions = []
    
    # Calculate number of complete shards in each dimension
    n_shards_h = height // shard_size
    n_shards_w = width // shard_size
    
    print(f"Will create {n_shards_h * n_shards_w} complete shards of size {shard_size}x{shard_size}")
    
    for i in range(0, height - shard_size + 1, shard_size):
        for j in range(0, width - shard_size + 1, shard_size):
            # Extract shard
            shard = data[i:i+shard_size, j:j+shard_size, :]
            shards.append(shard)
            positions.append((i, j))
            
    # Print information about the shards
    if len(shards) > 0:
        print(f"Individual shard shape: {shards[0].shape}")
    print(f"Number of shards created: {len(shards)}")
    
    return shards, positions

def reconstruct_from_shards(shards, positions, full_shape, shard_size=24):
    """
    Reconstruct full image from shards
    
    Args:
        shards: list of numpy arrays of shape (shard_size, shard_size, energy_points)
        positions: list of tuples containing (i, j) start positions
        full_shape: tuple of (height, width, energy_points)
        shard_size: size of each square shard
    
    Returns:
        reconstructed: numpy array of shape full_shape
    """
    height, width, energy_points = full_shape
    reconstructed = np.zeros(full_shape)
    
    for shard, (i, j) in zip(shards, positions):
        reconstructed[i:i+shard_size, j:j+shard_size, :] = shard
    
    return reconstructed

def process_data_with_vae(data, model, device, shard_size=24):
    """Process entire dataset through VAE using shards"""
    
    # Split adjusted data into shards
    shards, positions = split_into_shards(data, shard_size)
    reconstructed_shards = []
    
    print("Processing shards through VAE...")
    for shard in tqdm(shards):
        # Process through VAE
        with torch.no_grad():
            shard_tensor = torch.tensor(shard).unsqueeze(0).unsqueeze(0).float().to(device)
            mean, logvar = model.encode(shard_tensor)
            z = model.reparameterize(mean, logvar)
            reconstructed_shard = model.decode(z).squeeze().cpu().numpy()
            reconstructed_shards.append(reconstructed_shard)
    
    # Reconstruct full image using adjusted shape
    height, width, energy = data.shape
    adjusted_shape = (height, width, energy)
    reconstructed_data = reconstruct_from_shards(reconstructed_shards, positions, adjusted_shape)
    
    return reconstructed_data

def calculate_pcc(spec1, spec2, range_start=3000, range_end=1025):
    """Calculate Pearson Correlation Coefficient for spectra within specified range"""
    spec1_range = spec1[range_start:range_end]
    spec2_range = spec2[range_start:range_end]
    pcc, _ = pearsonr(spec1_range, spec2_range)
    return pcc

def calculate_pcc_map(data1, data2, range_start=900, range_end=1025):
    """Calculate PCC map for two datasets"""
    shape = data1.shape[:-1]
    pcc_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            pcc_map[i,j] = calculate_pcc(data1[i,j], data2[i,j], range_start, range_end)
        
    # Plot PCC maps
    plt.figure(figsize=(15, 6))
    plt.imshow(pcc_map, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.title('PCC: Anomalous vs VAE Reconstructed')
    
    plt.tight_layout()
    plt.savefig("2DMAP")

    # Create PCC histograms
    plt.figure(figsize=(10, 6))
    
    plt.hist(pcc_map.flatten(), 
             bins=50, alpha=0.5, label='Normal Pixels', density=True)    
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Density')
    plt.title('PCC Distribution: Normal vs Anomalous Pixels (VAE)')
    plt.legend()
    plt.savefig("HISTOGRAM")
    
    return pcc_map

def inject_peak_shift(data, shift_value):
    """Inject peak shift anomalies with specific shift value"""
    anomaly_specs = [
        # ('peak_shift', 500, 1, {'shift_value': shift_value})
        ('oxygen_deficiency', 6, 6, {'energy_range': energy_values,'deficiency_level': shift_value, 'ok_edge_start': 533})
    ]
    return inject_controlled_anomalies(data, anomaly_specs)

def process_with_pca(anomalous_data):
    """Process data with PCA and return the fitted PCA model"""
    SI_anomalous = hs.signals.Signal1D(anomalous_data)
    SI_anomalous.decomposition(algorithm="sklearn_pca")
    print("decomposition complete")
    return SI_anomalous 

def get_reconstruction(decomposed_signal, n_components):
    """Get reconstruction using specified number of components"""
    return decomposed_signal.get_decomposition_model(n_components).data

def threshold_otsu(image, bias_factor = 0.95):
    """
    Modified Otsu thresholding with bias factor to be more conservative
    
    Args:
        image: numpy array containing the image/data to be thresholded
        bias_factor: Factor to make threshold more conservative (>1.0 makes the threshold higher)
    Returns:
        float: optimal threshold value
    """
    # Flatten the image into 1D array
    pixels = image.flatten()
    
    # Get histogram
    hist, bin_edges = np.histogram(pixels, bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Calculate variance
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Apply bias to variance calculation
    variance12 = variance12 * (bin_centers[:-1] ** bias_factor)  # Weight higher values more
    
    # Find threshold that maximizes inter-class variance
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    
    return threshold

def calculate_detection_metrics(anomalous_data, reconstructed_data, 
                              anomaly_mask, threshold=None):
    """
    Calculate detection metrics including F1 score using Otsu's method for thresholding
    
    Args:
        original_data: Original EELS data
        anomalous_data: Data with injected anomalies
        reconstructed_data: Reconstructed data from model
        anomaly_mask: Binary mask showing true anomaly locations
        threshold: Optional manual threshold (if None, Otsu's method is used)
    
    Returns:
        dict: Dictionary containing metrics and threshold used
    """
    # Calculate PCC map
    print("Calculating the PCC map")
    pcc_map = calculate_pcc_map(anomalous_data, reconstructed_data)
    
    # Create mask for shard 19's region (assuming 24x24 shards)
    height, width = pcc_map.shape
    shard_size = 24
    shard_19_i = (18 // (width // shard_size)) * shard_size
    shard_19_j = (18 % (width // shard_size)) * shard_size
    
    # Create mask where True indicates pixels to include (everything except shard 19)
    valid_region_mask = np.ones_like(pcc_map, dtype=bool)
    valid_region_mask[shard_19_i:shard_19_i+shard_size, 
                     shard_19_j:shard_19_j+shard_size] = False
    
    # Create a masked PCC map excluding shard 19
    masked_pcc_map = pcc_map[valid_region_mask]
    
    # Use Otsu's method for thresholding if threshold not provided
    if threshold is None:
        # Convert PCCs to dissimilarity scores (1 - PCC), using only valid data
        dissimilarity = 1 - masked_pcc_map
        
        # Normalize to [0, 1] range for Otsu's method
        dissimilarity_norm = (dissimilarity - np.min(dissimilarity)) / \
                           (np.max(dissimilarity) - np.min(dissimilarity))
        
        # Get threshold using Otsu's method on masked data
        print("Getting threshold using Otsu's method (excluding shard 19)")
        thresh_otsu = threshold_otsu(dissimilarity_norm)
        
        # Convert threshold back to PCC scale
        threshold = 1 - (thresh_otsu * (np.max(dissimilarity) - np.min(dissimilarity)) + 
                        np.min(dissimilarity))
    
    print(f"Final PCC Threashold for anomaly detection: {threshold}")

     # Create predicted anomaly mask
    predicted_mask = pcc_map < threshold
    
    # Flatten masks for metric calculation, excluding shard 19's region
    y_true = anomaly_mask[valid_region_mask].flatten() > 0
    y_pred = predicted_mask[valid_region_mask].flatten()
    
    # Calculate metrics
    metrics = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'threshold': threshold,
        'pcc_map': pcc_map,
        'predicted_mask': predicted_mask,
        'confusion_matrix': {
            'true_positives': np.sum(np.logical_and(y_true, y_pred)),
            'false_positives': np.sum(np.logical_and(np.logical_not(y_true), y_pred)),
            'false_negatives': np.sum(np.logical_and(y_true, np.logical_not(y_pred))),
            'true_negatives': np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
        }
    }

    print(f"F1 score: {metrics['f1']}, Precision: {metrics['precision']}, Recall {metrics['recall']}") 
    return metrics

def run_experiment(original_data, model, device, shifts_range, pca_components=[3, 4, 5]):
    """Run experiment with improved metrics"""
    results = {
        'shifts': list(shifts_range),
        'pca': {n: [] for n in pca_components},
        'vae': [],
        'parameters': {
            'shifts_range': shifts_range,
            'pca_components': pca_components,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    }

    # Crop the spatial dimensions 
    cropped_original_data = original_data[:192, :192, :]
    print(f"Shape of the original data after cropping: {cropped_original_data.shape}")
    
    # Process each shift
    for shift in tqdm(shifts_range, desc="Processing shifts"):

        # Inject anomalies
        anomalous_data, anomaly_mask = inject_peak_shift(cropped_original_data, shift)
        print(f"Shape of the anomaly injected data is {anomalous_data.shape}")

        # Process with PCA
        decomposed_signal = process_with_pca(anomalous_data)
        
        # Get reconstructions and calculate metrics for each component number
        for n_components in pca_components:
            reconstructed_pca = get_reconstruction(decomposed_signal, n_components)
            metrics = calculate_detection_metrics(anomalous_data, reconstructed_pca, anomaly_mask)
            results['pca'][n_components].append(metrics)
        
        # Process with VAE
        reconstructed_vae = process_data_with_vae(anomalous_data, model, device)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)); ax1.imshow(np.sum(anomalous_data, axis=2)); ax2.imshow(np.sum(reconstructed_vae, axis=2)); ax1.set_title('Original'); ax2.set_title('VAE Reconstructed'); plt.savefig("DEBUG_IMAGE")
        metrics_vae = calculate_detection_metrics(anomalous_data, reconstructed_vae, anomaly_mask)
        results['vae'].append(metrics_vae)
    
    return results

def plot_results(results_file, save_path=None):
    """
    Plot results from saved JSON data file
    
    Args:
        results_file: Path to JSON results file
        save_path: Optional path to save the plot
    """
    # Load JSON data
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot PCA results
    for n_components, metrics in results['pca'].items():
        plt.plot(results['shifts'], 
                [m['f1'] for m in metrics],
                label=f'PCA ({n_components} components)', 
                marker='o', 
                markersize=4)
    
    # Plot VAE results
    plt.plot(results['shifts'], 
            [m['f1'] for m in results['vae']],
            label='VAE', 
            marker='o', 
            markersize=4, 
            linewidth=2, 
            color='red')
    
    plt.xlabel('Peak Shift (eV)')
    plt.ylabel('F1 Score')
    plt.title('Anomaly Detection Performance vs Peak Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def save_experimental_data(results, filename):
    """
    Save experimental data in JSON format
    
    Args:
        results: Dictionary containing experimental results
        filename: Base filename for saving data
    
    Returns:
        str: Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"{filename}_data_{timestamp}.json"
    
    # Prepare data for JSON serialization
    json_data = {
        'shifts': results['shifts'],
        'pca': {},
        'vae': [],
        'parameters': results['parameters']
    }
    
    # Process PCA results
    for n_components, metrics_list in results['pca'].items():
        json_data['pca'][str(n_components)] = []  # Convert key to string for JSON
        for metrics in metrics_list:
            # Only save numerical results and masks
            metric_data = {
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'threshold': metrics['threshold'],
                'predicted_mask': metrics['predicted_mask'],
                'pcc_map': metrics['pcc_map']
            }
            json_data['pca'][str(n_components)].append(metric_data)
    
    # Process VAE results
    for metrics in results['vae']:
        metric_data = {
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'threshold': metrics['threshold'],
            'predicted_mask': metrics['predicted_mask'],
            'pcc_map': metrics['pcc_map']
        }
        json_data['vae'].append(metric_data)
    
    # Save to JSON file
    with open(data_filename, 'w') as f:
        json.dump(json_data, f, cls=NumpyEncoder, indent=2)
    
    print(f"Data saved to {data_filename}")
    return data_filename 
   
def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shifts_range = np.arange(0.1, 1.0, 0.1)
    pca_components = [3, 4, 5]
    
    # Load your model
    from anomaly_detection.models.cvae3d import CVAE3D
    from anomaly_detection.config.config_handler import get_config
    
    print("Loading model and data...")

    config = get_config('/home/ssulta24/Desktop/VCAE_new/anomaly_detection/config/config.yaml')
    model = CVAE3D(config).to(device)
    model.load_state_dict(torch.load(
        '/home/ssulta24/Desktop/VCAE_new/wandb/run-20241006_160824-x3tu4esv/files/best_model.pth', 
        map_location=device
    ))
    model.eval()
    
    # Load your data
    from anomaly_detection.data.data_loader_V2 import EELSDataset
    data = EELSDataset(config)
    dataset = data.get_preprocessed_data()
    global energy_values
    energy_values = data.get_cropped_energy_values()

    print("Starting experiment...")

    # Run experiment
    results = run_experiment(dataset, model, device, shifts_range, pca_components)
    
    # Save data
    data_file = save_experimental_data(results, "anomaly_detection")
    
    # Create plot
    plot_path = f'anomaly_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plot_results(data_file, save_path=plot_path)

if __name__ == "__main__":
    main()