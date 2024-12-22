# anomaly_detection/main.py
# Author: Seyfal Sultanov 

import argparse
import logging
import os
import torch
import matplotlib.pyplot as plt
import wandb

from anomaly_detection.config.config_handler import get_config
from anomaly_detection.data.data_loader import get_data_loader
from anomaly_detection.training.train import train_model

from anomaly_detection.utils.utils import (
    visualize_inference,
    analyze_model_output,
    anomaly_detection_kde,
    visualize_anomalies,
    get_n_params
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="EELS Anomaly Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--mode", type=str, choices=['train', 'evaluate', 'detect'], required=True, help="Operation mode")
    return parser.parse_args()

def get_experiment_name():
    """Prompt the user to enter the name of the experiment."""
    return input("Enter the name of the experiment to restore the model from, or press enter for the default experiment: ")

def list_available_runs(project_name):
    """List available runs in the project."""
    api = wandb.Api()
    runs = api.runs(f"seyfal-university-of-illinois-chicago/{project_name}")
    print("Available runs:")
    for run in runs:
        print(f"- {run.name} (ID: {run.id})")

def get_run_id(project_name):
    """Prompt the user to enter the run ID."""
    list_available_runs(project_name)
    return input("Enter the run ID to restore the model from: ")

def load_model_from_wandb(model, config, mode):
    """Load the model from wandb based on user input."""
    experiment_input = get_experiment_name()
    if experiment_input == "": 
        experiment_name = config['experiment_name'] 
    else: 
        experiment_name = experiment_input

    run_id = get_run_id(config['wandb_project_name'])
    
    wandb.init(project=config['wandb_project_name'], name=experiment_name, id=run_id, resume="allow")
    
    try:
        best_model_path = wandb.restore('best_model.pth')
        model.load_state_dict(torch.load(best_model_path.name))
        model.eval()
        logger.info(f"Loaded best model from wandb (Run ID: {run_id})")
    except ValueError as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Please check the experiment name and run ID and try again.")
        wandb.finish()
        exit(1)
    
    return model

def main():
    args = parse_arguments()
    config = get_config(args.config)

    if args.mode == 'train':
        # Train the model
        logger.info("Starting model training")
        logger.info("Press 'X' and Enter at any time to stop training and save the model")
        trained_model = train_model(config)
        logger.info("Model training completed")

    elif args.mode in ['evaluate', 'detect']:
        # Load the model from wandb
        model = load_model_from_wandb(model, config, args.mode)
        logger.info(f"Model imported with {get_n_params(model)} parameters")

        # Set device
        device = torch.device(config['device'])
        logger.info(f"Using device: {device}")

        # Create data loader
        data_loader = get_data_loader(config)
        logger.info("Data loader created")

        if args.mode == 'evaluate':
            # Get a batch of data for evaluation
            eval_data = next(iter(data_loader))
            
            # Visualize inference
            logger.info("Visualizing inference")
            fig = visualize_inference(model, eval_data[0], eval_data[0], config)
            wandb.log({"inference_visualization": wandb.Image(fig)})

            # Analyze model output
            logger.info("Analyzing model output")
            with torch.no_grad():
                output, _, _ = model(eval_data.to(device))
            fig = analyze_model_output(eval_data[0].numpy(), output[0].cpu().numpy(), config)
            wandb.log({"model_output_analysis": wandb.Image(fig)})

        elif args.mode == 'detect':
            # Get a batch of data for anomaly detection
            detect_data = next(iter(data_loader))

            # Perform anomaly detection
            logger.info("Performing anomaly detection")
            anomaly_scores, anomaly_mask = anomaly_detection_kde(model, detect_data[0].numpy(), config)

            # Visualize anomalies
            logger.info("Visualizing anomalies")
            fig = visualize_anomalies(detect_data[0].numpy(), anomaly_scores, anomaly_mask)
            wandb.log({"anomaly_detection": wandb.Image(fig)})

        wandb.finish()

    logger.info("Script execution completed")

if __name__ == "__main__":
    main()