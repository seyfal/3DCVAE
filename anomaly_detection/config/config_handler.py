# anomaly_detection/config/config_handler.py
# Author: Seyfal Sultanov 

import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing the configuration.

    Raises:
        ConfigurationError: If the file is not found or there's an error in parsing.
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML file: {e}")

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration to ensure all required fields are present.

    Args:
        config (Dict[str, Any]): The configuration dictionary to validate.

    Raises:
        ConfigurationError: If a required field is missing or has an invalid value.
    """
    required_fields = [
        'data_path', 'energy_range', 'batch_size', 'num_workers', 'device',
        'latent_dim', 'learning_rate', 'lr_patience', 'lr_factor', 'kl_weight',
        'clip_value', 'epochs', 'save_interval', 'experiment_name'
    ]

    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required configuration field: {field}")

    # Add any specific validation rules here
    if config['batch_size'] <= 0:
        raise ConfigurationError("batch_size must be a positive integer")

    if not os.path.exists(config['data_path']):
        raise ConfigurationError(f"Data path does not exist: {config['data_path']}")

    logger.info("Configuration validated successfully")

def get_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate the configuration.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Validated configuration dictionary.

    Raises:
        ConfigurationError: If there's an error in loading or validating the configuration.
    """
    config = load_config(config_path)
    validate_config(config)
    return config