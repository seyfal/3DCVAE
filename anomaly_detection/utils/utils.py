# anomaly_detection/utils/utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import scoreatpercentile
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

def visualize_inference(model, actual_image, preprocessed_image, config):
    """
    Visualize the actual image, preprocessed input, and model prediction for EELS data.
    
    Args:
        model (CVAE3D): The trained CVAE3D model
        actual_image (numpy.ndarray): The original 3D EELS image
        preprocessed_image (numpy.ndarray): The preprocessed 3D image used as input to the model
        config (dict): Configuration dictionary
    """
    model.eval()
    device = torch.device(config['device'])
    energy_range = config['energy_range']
    pixel_x = config.get('visualization_pixel_x', preprocessed_image.shape[0] // 2)
    pixel_y = config.get('visualization_pixel_y', preprocessed_image.shape[1] // 2)

    with torch.no_grad():
        if preprocessed_image.ndim == 4:  # If it's already 4D (C, D, H, W)
            input_tensor = torch.tensor(preprocessed_image).unsqueeze(0).float().to(device)
        elif preprocessed_image.ndim == 3:  # If it's 3D (D, H, W)
            input_tensor = torch.tensor(preprocessed_image).unsqueeze(0).unsqueeze(0).float().to(device)
        else:
            raise ValueError(f"Unexpected input shape: {preprocessed_image.shape}")

        mean, logvar = model.encode(input_tensor)
        z = model.reparameterize(mean, logvar)
        prediction = model.decode(z)
        prediction_np = prediction.squeeze().cpu().numpy()

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    
    start_pixel = int((energy_range[0] - 0))
    end_pixel = int((energy_range[1] - 0))
    actual_image = actual_image[:, :, start_pixel:end_pixel]

    images = [actual_image, preprocessed_image, prediction_np]
    titles = ['Actual Image', 'Preprocessed Input', 'Model Prediction']

    for i, (img, title) in enumerate(zip(images, titles)):
        spatial_img = np.sum(img, axis=2)
        im = axs[i, 0].imshow(spatial_img, cmap='viridis')
        axs[i, 0].set_title(f'{title} (Sum along energy axis)')
        axs[i, 0].set_xlabel('X axis')
        axs[i, 0].set_ylabel('Y axis')
        axs[i, 0].plot(pixel_x, pixel_y, 'r+', markersize=10)
        plt.colorbar(im, ax=axs[i, 0])

    spectra = [actual_image[pixel_x, pixel_y, :],
               preprocessed_image[pixel_x, pixel_y, :],
               prediction_np[pixel_x, pixel_y, :]]

    for i, (spectrum, title) in enumerate(zip(spectra, titles)):
        energy_values = np.linspace(energy_range[0], energy_range[1], len(spectrum))
        axs[i, 1].plot(energy_values, spectrum)
        axs[i, 1].set_title(f'{title} Spectrum at pixel ({pixel_x}, {pixel_y})')
        axs[i, 1].set_xlabel('Energy (eV)')
        axs[i, 1].set_ylabel('Intensity')

    plt.tight_layout()
    plt.show()

def analyze_model_output(input_data, output_data, config):
    """
    Analyze the difference between model input and output.
    
    Args:
        input_data (numpy.ndarray): The input data to the model
        output_data (numpy.ndarray): The output data from the model
        config (dict): Configuration dictionary
    """
    assert input_data.shape == output_data.shape, "Input and output shapes do not match"
    
    energy_range = config['energy_range']
    num_random_spectra = config.get('num_random_spectra', 5)
    
    error = np.square(input_data - output_data)
    ssim_value = structural_similarity(input_data, output_data, data_range=input_data.max() - input_data.min())
    psnr_value = peak_signal_noise_ratio(input_data, output_data, data_range=input_data.max() - input_data.min())
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    axs[0, 0].imshow(np.sum(input_data, axis=2), cmap='viridis')
    axs[0, 0].set_title('Input (Sum along energy axis)')
    axs[0, 1].imshow(np.sum(output_data, axis=2), cmap='viridis')
    axs[0, 1].set_title('Output (Sum along energy axis)')
    
    im = axs[1, 0].imshow(np.sum(error, axis=2), cmap='hot')
    axs[1, 0].set_title('Error Heatmap (Sum of squared errors)')
    plt.colorbar(im, ax=axs[1, 0])
    
    axs[1, 1].set_title(f'Random Spectra Comparison (SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f})')
    energy_values = np.linspace(energy_range[0], energy_range[1], input_data.shape[2])
    
    for _ in range(num_random_spectra):
        x, y = np.random.randint(0, input_data.shape[0]), np.random.randint(0, input_data.shape[1])
        axs[1, 1].plot(energy_values, input_data[x, y, :], 'b-', alpha=0.5)
        axs[1, 1].plot(energy_values, output_data[x, y, :], 'r-', alpha=0.5)
    
    axs[1, 1].set_xlabel('Energy (eV)')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].legend(['Input', 'Output'])
    
    plt.tight_layout()
    plt.show()
    
    logger.info(f"SSIM: {ssim_value:.4f}")
    logger.info(f"PSNR: {psnr_value:.4f}")

def anomaly_detection_kde(model, data, config):
    """
    Perform anomaly detection using Kernel Density Estimation on the latent space.
    
    Args:
        model (CVAE3D): Trained CVAE3D model
        data (numpy.ndarray): Input data of shape (height, width, energy)
        config (dict): Configuration dictionary
    
    Returns:
        numpy.ndarray: Anomaly scores for each pixel
        numpy.ndarray: Boolean mask of anomalies
    """
    model.eval()
    device = torch.device(config['device'])
    threshold_percentile = config.get('anomaly_threshold_percentile', 1)
    bandwidth = config.get('kde_bandwidth', 1.0)
    
    height, width, energy = data.shape
    data_tensor = torch.tensor(data.reshape(1, 1, height, width, energy), dtype=torch.float32).to(device)

    with torch.no_grad():
        mean, _ = model.encode(data_tensor)
    latent_representation = mean.cpu().numpy()

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(latent_representation)

    log_density = kde.score_samples(latent_representation)
    anomaly_score = -log_density[0]

    with torch.no_grad():
        reconstructed = model.decode(mean)
    
    reconstruction_error = torch.abs(data_tensor - reconstructed).mean(dim=-1).squeeze().cpu().numpy()

    anomaly_scores = reconstruction_error * anomaly_score

    threshold = scoreatpercentile(anomaly_scores, 100 - threshold_percentile)
    anomalies = anomaly_scores > threshold

    return anomaly_scores, anomalies

def visualize_anomalies(original_data, anomaly_scores, anomaly_mask):
    """
    Visualize the original data, anomaly scores, and anomaly mask.
    
    Args:
        original_data (numpy.ndarray): Original input data
        anomaly_scores (numpy.ndarray): Computed anomaly scores
        anomaly_mask (numpy.ndarray): Boolean mask of detected anomalies
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(np.mean(original_data, axis=-1), cmap='viridis')
    plt.title('Original Data (Mean Intensity)')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(anomaly_scores, cmap='hot_r')
    plt.title('Anomaly Scores')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(anomaly_mask, cmap='binary')
    plt.title('Anomaly Mask')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def get_n_params(model):
    """
    Get the number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        int: Number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_reconstructions(model, data_loader, device, n_samples=10):
    """
    Visualize original and reconstructed images.
    
    Args:
        model (CVAE3D): Trained CVAE3D model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to run the model on
        n_samples (int): Number of samples to visualize
    
    Returns:
        plt.Figure: Figure containing the original and reconstructed images
    """
    model.eval()
    originals = []
    reconstructions = []

    with torch.no_grad():
        for batch in data_loader:
            if len(originals) >= n_samples:
                break
            x = batch.to(device)
            recon_x, _, _ = model(x)
            originals.extend(x.cpu().numpy())
            reconstructions.extend(recon_x.cpu().numpy())

    originals = np.array(originals[:n_samples])
    reconstructions = np.array(reconstructions[:n_samples])

    # Create plot
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5*n_samples))
    for i in range(n_samples):
        axes[i, 0].imshow(np.sum(originals[i, 0], axis=-1), cmap='viridis')
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.sum(reconstructions[i, 0], axis=-1), cmap='viridis')
        axes[i, 1].set_title(f"Reconstruction {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    return fig

def display_data_info(dataset, dataloader):
    """
    Display information about the EELS dataset and dataloader.
    
    Args:
    dataset (EELSDataset): The EELS dataset
    dataloader (DataLoader): The EELS dataloader
    """
    full_image = dataset.get_full_image()
    print(f"Original data shape: {dataset.signal.data.shape}")
    print(f"Preprocessed full image shape: {full_image.shape}")
    print(f"Number of sub-images: {len(dataset)}")
    print(f"Sub-image shape: {dataset[0].shape}")
    print(f"Spatial dimensions: {full_image.shape[0]}x{full_image.shape[1]}")
    print(f"Energy range: {dataset.energy_range}")
    print(f"Batch size: {dataloader.batch_size}")

def plot_eels_spectrum(dataset, spatial_coord=(0, 0)):
    """
    Plot the EELS spectrum for a given spatial coordinate.
    
    Args:
    dataset (EELSDataset): The EELS dataset
    spatial_coord (tuple): The (x, y) coordinate of the pixel to plot
    """
    x, y = spatial_coord
    full_image = dataset.get_full_image()
    spectrum = full_image[y, x, :]
    energy_values = np.linspace(dataset.energy_range[0], dataset.energy_range[1], len(spectrum))
    
    plt.figure(figsize=(10, 6))
    plt.plot(energy_values, spectrum)
    plt.title(f"EELS Spectrum at pixel ({x}, {y})")
    plt.xlabel("Energy Loss (eV)")
    plt.ylabel("Intensity")
    plt.show()

def interactive_eels_visualization(dataset):
    """
    Create an interactive visualization for EELS data.
    
    Args:
    dataset (EELSDataset): The EELS dataset
    """
    full_image = dataset.get_full_image()
    energy_values = np.linspace(dataset.energy_range[0], dataset.energy_range[1], full_image.shape[2])
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # Plot sum image
    sum_img = axs[0].imshow(np.sum(full_image, axis=2), cmap='viridis')
    axs[0].set_title('Sum Image')
    
    # Create empty line plots for spectra
    spectrum, = axs[1].plot(energy_values, np.zeros_like(energy_values))
    axs[1].set_title('EELS Spectrum')
    axs[1].set_xlabel('Energy Loss (eV)')
    axs[1].set_ylabel('Intensity')
    
    # Create 2D heatmap
    heatmap = axs[2].imshow(full_image[:, :, 0], cmap='viridis')
    axs[2].set_title('2D Heatmap')
    
    # Create a slider for adjusting the energy slice
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(ax_slider, 'Energy Slice', 0, full_image.shape[2]-1, valinit=0, valstep=1)
    
    # Function to update heatmap based on slider
    def update_slice(val):
        slice_index = int(slider.val)
        heatmap.set_array(full_image[:, :, slice_index])
        axs[2].set_title(f'2D Heatmap at {energy_values[slice_index]:.2f} eV')
        fig.canvas.draw_idle()
    
    slider.on_changed(update_slice)
    
    # Function to update spectrum on mouse hover
    def update_spectrum(event):
        if event.inaxes in [axs[0], axs[2]]:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < full_image.shape[1] and 0 <= y < full_image.shape[0]:
                spectrum.set_ydata(full_image[y, x, :])
                axs[1].relim()
                axs[1].autoscale_view()
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', update_spectrum)
    
    plt.show()

def get_tensor_info(tensor):
    """
    Display information about a PyTorch tensor.
    
    Args:
    tensor (torch.Tensor): The tensor to analyze
    """
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor min value: {tensor.min().item()}")
    print(f"Tensor max value: {tensor.max().item()}")
    print(f"Tensor mean value: {tensor.mean().item()}")
    print(f"Tensor standard deviation: {tensor.std().item()}")

def visualize_batch(batch, energy_range):
    """
    Visualize a batch of EELS data.
    
    Args:
    batch (torch.Tensor): A batch of EELS data (shape: batch_size x 1 x height x width x energy)
    energy_range (tuple): The energy range (start, end) in eV
    """
    batch_size, _, height, width, energy = batch.shape
    energy_values = np.linspace(energy_range[0], energy_range[1], energy)
    
    fig, axs = plt.subplots(2, batch_size, figsize=(5*batch_size, 10))
    
    for i in range(batch_size):
        # Plot sum image
        axs[0, i].imshow(torch.sum(batch[i, 0], dim=2).cpu(), cmap='viridis')
        axs[0, i].set_title(f'Sum Image {i}')
        axs[0, i].axis('off')
        
        # Plot center spectrum
        center_x, center_y = width // 2, height // 2
        spectrum = batch[i, 0, center_y, center_x, :].cpu()
        axs[1, i].plot(energy_values, spectrum)
        axs[1, i].set_title(f'Center Spectrum {i}')
        if i == 0:
            axs[1, i].set_ylabel('Intensity')
        if i == batch_size - 1:
            axs[1, i].set_xlabel('Energy Loss (eV)')
    
    plt.tight_layout()
    plt.show()

def visualize_shard(dataset, shard_idx):
    """
    Visualize a specific shard from the dataset.
    
    Args:
    dataset (EELSDataset): The EELS dataset
    shard_idx (int): Index of the shard to visualize
    """
    shard = dataset.get_shard(shard_idx)
    energy_values = np.linspace(dataset.energy_range[0], dataset.energy_range[1], shard.shape[-1])
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot sum image
    axs[0].imshow(np.sum(shard[0], axis=2), cmap='viridis')
    axs[0].set_title(f'Sum Image (Shard {shard_idx})')
    axs[0].axis('off')
    
    # Plot center spectrum
    center_x, center_y = shard.shape[2] // 2, shard.shape[3] // 2
    spectrum = shard[0, 0, center_y, center_x, :]
    axs[1].plot(energy_values, spectrum)
    axs[1].set_title(f'Center Spectrum (Shard {shard_idx})')
    axs[1].set_xlabel('Energy Loss (eV)')
    axs[1].set_ylabel('Intensity')
    
    # Plot 2D heatmap of central energy
    central_energy = shard.shape[-1] // 2
    axs[2].imshow(shard[0, 0, :, :, central_energy], cmap='viridis')
    axs[2].set_title(f'2D Heatmap at {energy_values[central_energy]:.2f} eV')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# dataset = EELSDataset(config)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# display_data_info(dataset, dataloader)
# plot_eels_spectrum(dataset, spatial_coord=(50, 50))
# interactive_eels_visualization(dataset)
# 
# # Get a batch
# batch = next(iter(dataloader))
# get_tensor_info(batch)
# visualize_batch(batch, dataset.energy_range)
# 
# # Visualize a specific shard
# visualize_shard(dataset, shard_idx=0)

def visualize_preprocessed_eels(dataset, pixel_x=None, pixel_y=None):
    """
    Visualize the preprocessed EELS image by showing a 2D spatial image and the spectrum for a selected pixel.

    Args:
    dataset (EELSDataset): The EELS dataset
    pixel_x (int, optional): X coordinate of the pixel to show spectrum. If None, the center pixel is used.
    pixel_y (int, optional): Y coordinate of the pixel to show spectrum. If None, the center pixel is used.

    Returns:
    None (displays the plot)
    """
    # Get the full preprocessed image
    preprocessed_img = dataset.get_full_image()

    # Ensure the input is 3D
    assert len(preprocessed_img.shape) == 3, "Input should be a 3D array"

    # If pixel coordinates are not provided, use the center pixel
    if pixel_x is None:
        pixel_x = preprocessed_img.shape[0] // 2
    if pixel_y is None:
        pixel_y = preprocessed_img.shape[1] // 2

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 2D spatial image (sum along energy axis)
    spatial_img = np.sum(preprocessed_img, axis=2)
    im = ax1.imshow(spatial_img, cmap='viridis')
    ax1.set_title('2D Spatial Image (Sum along energy axis)')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.plot(pixel_x, pixel_y, 'r+', markersize=10)  # Mark the selected pixel
    plt.colorbar(im, ax=ax1)

    # Plot spectrum for the selected pixel
    spectrum = preprocessed_img[pixel_x, pixel_y, :]
    energy_values = np.linspace(dataset.energy_range[0], dataset.energy_range[1], len(spectrum))
    ax2.plot(energy_values, spectrum)
    ax2.set_title(f'Spectrum at pixel ({pixel_x}, {pixel_y})')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Intensity (normalized)')

    plt.tight_layout()
    plt.show()

# Example usage:
# dataset = EELSDataset(config)
# visualize_preprocessed_eels(dataset)
# visualize_preprocessed_eels(dataset, pixel_x=100, pixel_y=100)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_sub_images(dataset, max_samples=36, anomaly_threshold=2, figsize=(20, 20)):
    """
    Visualize multiple sub-images from the EELSDataset to identify anomalies between samples.

    Parameters:
    - dataset: EELSDataset object
    - max_samples: Maximum number of sub-images to visualize (default is 36)
    - anomaly_threshold: Number of standard deviations from mean to consider as anomaly
    - figsize: Figure size for the plot

    Returns:
    - None (displays the plot)
    """
    # Determine the total number of shards in the dataset
    total_shards = len(dataset)
    
    # Determine the number of samples to visualize
    num_samples = min(total_shards, max_samples)
    
    # Get multiple sub-images (shards) from the dataset
    data = np.array([dataset.get_shard(i)[0] for i in range(num_samples)])
    
    X, Y, Y, Z = data.shape
    # Calculate the number of rows and columns for subplots
    n_rows = int(np.ceil(np.sqrt(X)))
    n_cols = int(np.ceil(X / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f"Sample Visualization with Anomaly Detection (Total Shards: {total_shards})", fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Calculate global mean and std for anomaly detection
    global_mean = np.mean(data)
    global_std = np.std(data)
    
    for i in range(X):
        ax = axes[i]
        sample = data[i]
        
        # Sum along the spectral dimension
        image = np.sum(sample, axis=-1)
        
        # Detect anomalies
        anomaly_mask = np.abs(image - global_mean) > anomaly_threshold * global_std
        
        # Create a color overlay for anomalies
        overlay = np.zeros((*image.shape, 4))
        overlay[anomaly_mask] = [1, 0, 0, 0.5]  # Red with 50% opacity
        
        # Plot the image
        im = ax.imshow(image, cmap='viridis')
        ax.imshow(overlay)
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Remove any unused subplots
    for i in range(X, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Example usage:
# dataset = EELSDataset(config)
# visualize_tensor_anomalies(dataset)