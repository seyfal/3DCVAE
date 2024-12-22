# anomaly_detection/data/data_loader.py
# Author: Seyfal Sultanov 

import numpy as np
import hyperspy.api as hs
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset, DataLoader
import torch
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class EELSDataset(Dataset):
    """
    Dataset class for EELS (Electron Energy Loss Spectroscopy) data.
    
    This class handles loading, preprocessing, and providing access to EELS data.
    """

    def __init__(self, config):
        """
        Initialize the EELSDataset.

        Args:
            config (json): settings from anomaly_detection/config/config.yaml
        """
        self.filepath = config['data_path']
        self.energy_range = tuple(config['energy_range'])
        self.size = config['image_size']
        self.sigma = config['sigma']
        self.xy_window = config['xy_window']
        self.stride = config['sliding_window_stride']
        self.anomaly_indices = config.get('anomaly_indices', [])
        self.scaling_method = config.get('scaling_method', 'robust_minmax')
        self.scaling_params = None 
        self.pre_scaled_anomalous_sub_images = None
        self.post_scaled_anomalous_sub_images = None

        self.signal = self.load_dm_data()
        self.original_data = self.signal.data
        self.calibrated_data, self.energy_values = self.extract_calibrated_data()
        self.cropped_data = self.crop_energy_range()
        self.preprocessed_data = self.preprocess_full_image()
        self.training_sub_images, self.anomaly_sub_images = self.generate_sub_images()

    def load_dm_data(self):
        try:
            signal = hs.load(self.filepath)
            logger.info(f"Successfully loaded data from {self.filepath} with shape {signal.data.shape}")
            return signal
        except Exception as e:
            logger.error(f"Error loading data from {self.filepath}: {str(e)}")
            raise

    def extract_calibrated_data(self):
        try:
            energy_axis = self.signal.axes_manager[-1]
            energy_values = np.arange(energy_axis.size) * energy_axis.scale + energy_axis.offset
            logger.info(f"Original data shape: {self.original_data.shape}")
            logger.info(f"Energy range: {energy_values[0]} to {energy_values[-1]} {energy_axis.units}")
        except AttributeError:
            logger.warning("Couldn't extract energy axis information. Using index for energy axis.")
            energy_values = np.arange(self.original_data.shape[-1])

        return self.original_data, energy_values

    def crop_energy_range(self):
        if self.energy_range:
            start_idx = np.searchsorted(self.energy_values, self.energy_range[0])
            end_idx = np.searchsorted(self.energy_values, self.energy_range[1])
            
            # Calculate the number of points in the range
            num_points = end_idx - start_idx
            
            # Find the closest number of points divisible by 8
            # target_points = round(num_points / 32) * 32
            target_points = 1312

            # Adjust start and end indices to center the range
            mid_idx = (start_idx + end_idx) // 2
            half_target = target_points // 2
            new_start_idx = max(0, mid_idx - half_target)
            new_end_idx = min(len(self.energy_values), new_start_idx + target_points)
            
            # If we hit the upper bound, adjust the start index
            if new_end_idx == len(self.energy_values):
                new_start_idx = new_end_idx - target_points
            
            cropped_data = self.calibrated_data[..., new_start_idx:new_end_idx]
            self.cropped_energy_values = self.energy_values[new_start_idx:new_end_idx]
            
            logger.info(f"Original energy range: {self.energy_values[start_idx]} to {self.energy_values[end_idx-1]} eV")
            logger.info(f"Adjusted energy range: {self.cropped_energy_values[0]} to {self.cropped_energy_values[-1]} eV")
            logger.info(f"Number of energy points: {len(self.cropped_energy_values)} (divisible by 32)")
        else:
            # If no energy range is specified, ensure the full range is divisible by 8
            num_points = len(self.energy_values)
            target_points = (num_points // 32) * 32
            cropped_data = self.calibrated_data[..., :target_points]
            self.cropped_energy_values = self.energy_values[:target_points]
            
            logger.info(f"No energy range specified. Using full range: {self.cropped_energy_values[0]} to {self.cropped_energy_values[-1]} eV")
            logger.info(f"Number of energy points: {len(self.cropped_energy_values)} (divisible by 32)")

        logger.info(f"Cropped data shape: {cropped_data.shape}")
        return cropped_data

    def preprocess_full_image(self):
        # Apply Gaussian blur
        blurred_image = gaussian(self.cropped_data, sigma=self.sigma, mode='reflect', preserve_range=True)
        
        # Save pre-scaled anomalous sub-images
        self.save_pre_scaled_anomalous_sub_images(blurred_image)
        
        if self.scaling_method == 'robust_minmax':
            scaled_image, self.scaling_params = self.robust_minmax_scaling(blurred_image)
        elif self.scaling_method == 'zscore':
            scaled_image, self.scaling_params = self.zscore_scaling(blurred_image)
        elif self.scaling_method == 'quantile':
            scaled_image, self.scaling_params = self.quantile_scaling(blurred_image)
        elif self.scaling_method == 'minmax':
            scaled_image, self.scaling_params = self.minmax_scaling(blurred_image)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        # Scale anomalous sub-images separately
        self.scale_anomalous_sub_images()
        
        # Apply spatial-spectral smoothing
        smoothed_img = self.smooth_spatial_spectral(scaled_image)
        
        logger.info(f"Preprocessed data shape: {smoothed_img.shape}")
        logger.info(f"Number of pre-scaled anomalous sub-images: {len(self.pre_scaled_anomalous_sub_images)}")
        
        return smoothed_img.astype('float32')

    def minmax_scaling(self, data):
        all_sub_images = self.sliding_window(data)
        non_anomalous_sub_images = [img for i, img in enumerate(all_sub_images) if i not in self.anomaly_indices]
        
        min_val = np.min(non_anomalous_sub_images)
        max_val = np.max(non_anomalous_sub_images)
        scaled_image = (data - min_val) / (max_val - min_val)
        return scaled_image, {'min_val': min_val, 'max_val': max_val}

    def robust_minmax_scaling(self, data):
        # q_low, q_high = np.percentile(data, [0.005, 99.995]) # these are the correct ratios 
        q_low, q_high = np.percentile(data, [1, 99]) # these are ratios that the model was trained on 
        scaled_data = (data - q_low) / (q_high - q_low)
        scaled_data = np.clip(scaled_data, 0, 1)
        return scaled_data, {'q_low': q_low, 'q_high': q_high}

    def zscore_scaling(self, data):
        mean = np.mean(data)
        std = np.std(data)
        scaled_data = (data - mean) / std
        return scaled_data, {'mean': mean, 'std': std}

    def quantile_scaling(self, data):
        flat_data = data.flatten()
        ranks = stats.rankdata(flat_data)
        scaled_data = ranks.reshape(data.shape) / len(flat_data)
        return scaled_data, {'original_shape': data.shape}
    
    def scale_anomalous_sub_images(self):
        if self.pre_scaled_anomalous_sub_images is not None:
            if self.scaling_method == 'robust_minmax':
                scaled, _ = self.robust_minmax_scaling(self.pre_scaled_anomalous_sub_images)
            elif self.scaling_method == 'zscore':
                scaled, _ = self.zscore_scaling(self.pre_scaled_anomalous_sub_images)
            elif self.scaling_method == 'quantile':
                scaled, _ = self.quantile_scaling(self.pre_scaled_anomalous_sub_images)
            elif self.scaling_method == 'minmax':
                scaled, _ = self.minmax_scaling(self.pre_scaled_anomalous_sub_images)
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            self.post_scaled_anomalous_sub_images = scaled
            logger.info(f"Scaled {len(self.post_scaled_anomalous_sub_images)} anomalous sub-images")

    def save_pre_scaled_anomalous_sub_images(self, data):
        if self.pre_scaled_anomalous_sub_images is None:
            all_sub_images = self.sliding_window(data)
            self.pre_scaled_anomalous_sub_images = np.array([img for i, img in enumerate(all_sub_images) if i in self.anomaly_indices])
            logger.info(f"Saved {len(self.pre_scaled_anomalous_sub_images)} pre-scaled anomalous sub-images")

    def get_pre_scaled_anomalous_sub_images(self):
        if self.pre_scaled_anomalous_sub_images is None:
            logger.warning("No pre-scaled anomalous sub-images have been saved yet.")
            return None
        return self.pre_scaled_anomalous_sub_images
    
    def get_post_scaled_anomalous_sub_images(self):
        if self.post_scaled_anomalous_sub_images is None:
            logger.warning("No post-scaled anomalous sub-images have been saved yet.")
            return None
        return self.post_scaled_anomalous_sub_images

    def generate_sub_images(self):
        all_sub_images = self.sliding_window(self.preprocessed_data)
        
        training_sub_images = []
        anomaly_sub_images = []
        
        for i, sub_image in enumerate(all_sub_images):
            if i in self.anomaly_indices:
                anomaly_sub_images.append(sub_image)
            else:
                training_sub_images.append(sub_image)
        
        training_sub_images = np.array(training_sub_images)
        anomaly_sub_images = np.array(anomaly_sub_images)
        
        # Reshape sub-images for PyTorch (batch_size, channels, height, width, energy)
        training_sub_images = training_sub_images.reshape(-1, 1, self.size, self.size, training_sub_images.shape[-1])
        anomaly_sub_images = anomaly_sub_images.reshape(-1, 1, self.size, self.size, anomaly_sub_images.shape[-1])
        
        logger.info(f"Generated {len(training_sub_images)} training sub-images")
        logger.info(f"Generated {len(anomaly_sub_images)} anomalous sub-images")
        
        return training_sub_images, anomaly_sub_images
    
    def smooth_spatial_spectral(self, arr):
        neighborhood_sum = uniform_filter(arr, size=(self.xy_window, self.xy_window, 1), mode='reflect')
        neighborhood_count = uniform_filter(np.ones_like(arr), size=(self.xy_window, self.xy_window, 1), mode='reflect')
        return neighborhood_sum / neighborhood_count

    def sliding_window(self, data):
        height, width, energy = data.shape
        sub_images = []
        
        for i in range(0, height - self.size + 1, self.stride):
            for j in range(0, width - self.size + 1, self.stride):
                sub_image = data[i:i+self.size, j:j+self.size, :]
                sub_images.append(sub_image)
        
        return np.array(sub_images)
    
    def reverse_scaling(self, scaled_data):
        if self.scaling_method == 'robust_minmax':
            return self.reverse_robust_minmax(scaled_data)
        elif self.scaling_method == 'zscore':
            return self.reverse_zscore(scaled_data)
        elif self.scaling_method == 'quantile':
            return self.reverse_quantile(scaled_data)
        elif self.scaling_method == 'minmax':
            return self.reverse_minmax(scaled_data)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def reverse_robust_minmax(self, scaled_data):
        q_low, q_high = self.scaling_params['q_low'], self.scaling_params['q_high']
        return scaled_data * (q_high - q_low) + q_low

    def reverse_zscore(self, scaled_data):
        mean, std = self.scaling_params['mean'], self.scaling_params['std']
        return scaled_data * std + mean

    def reverse_quantile(self, scaled_data):
        # Note: This is an approximation, as exact reversal is not possible
        original_shape = self.scaling_params['original_shape']
        return stats.norm.ppf(scaled_data).reshape(original_shape)

    def reverse_minmax(self, scaled_data):
        min_val, max_val = self.scaling_params['min_val'], self.scaling_params['max_val']
        return scaled_data * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.training_sub_images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.training_sub_images[idx])

    # Methods to access data at different stages
    def get_original_data(self):
        return self.original_data

    def get_calibrated_data(self):
        return self.calibrated_data

    def get_cropped_data(self):
        return self.cropped_data

    def get_preprocessed_data(self):
        return self.preprocessed_data
    
    def get_training_sub_images(self):
        return self.training_sub_images

    def get_anomaly_sub_images(self):
        return self.anomaly_sub_images

    def get_anomaly_indices(self):
        return self.anomaly_indices

    def get_sub_images(self):
        return self.sub_images

    def get_energy_values(self):
        return self.energy_values

    def get_cropped_energy_values(self):
        return self.cropped_energy_values

    def get_scaling_params(self):
        return self.scaling_method, self.scaling_params

def get_data_loader(config):
    try:
        dataset = EELSDataset(config)
        if len(dataset) == 0:
            logger.warning("Dataset is empty. Check your data and preprocessing steps.")
            return None
        return DataLoader(dataset, 
                          batch_size=config['batch_size'], 
                          shuffle=True, 
                          num_workers=config['num_workers'])
    except Exception as e:
        logger.error(f"Error creating DataLoader: {str(e)}")
        return None
