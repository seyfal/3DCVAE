# anomaly_detection/training/train.py
# Author: Seyfal Sultanov 

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
import wandb
import sys
import select
import numpy as np
import base64
import io
import os
from PIL import Image
from torchinfo import summary
import matplotlib as plt

from anomaly_detection.models.cvae3d import CVAE3D
from anomaly_detection.data.data_loader import get_data_loader
from anomaly_detection.utils.utils import visualize_reconstructions

logger = logging.getLogger(__name__)

def compute_loss(model, x, kl_weight):
    """Compute the loss for the CVAE3D model."""
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    reconstruction_loss = F.mse_loss(x_logit, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    total_loss = reconstruction_loss + kl_weight * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss

def compute_elbo_loss(model, x, kl_weight=1.0, sam_weight=2.0, auc_weight=1.0):
    """
    Compute the ELBO loss for the VAE model with additional pixel-wise AUC comparison.
    
    Args:
        model: The VAE model
        x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width, spectrum_length)
        kl_weight (float): Weight for the KL divergence term
        sam_weight (float): Weight for the SAM loss term
        auc_weight (float): Weight for the AUC comparison term
    
    Returns:
        tuple: Contains total_loss, elbo, recon_loss, kl_loss, sam, auc_loss
    """
    # Encode the input
    mean, logvar = model.encode(x)
    
    # Sample from the latent space
    z = model.reparameterize(mean, logvar)
    
    # Decode the latent sample
    x_recon = model.decode(z)
    
    # # Reconstruction loss (negative log-likelihood)
    # recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # Ensure x_logit is in log-probability space
    x_log_prob = F.log_softmax(x_recon, dim=1)
    
    # Flatten the input and target tensors
    x_flat = x.view(-1, x.size(-1))
    x_log_prob_flat = x_log_prob.view(-1, x_log_prob.size(-1))
    
    # Compute negative log-likelihood loss
    recon_loss = F.nll_loss(x_log_prob_flat, x_flat.argmax(dim=1), reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # # Spectral Angle Mapper (SAM) loss
    # def sam_loss(y_true, y_pred):
    #     dot_product = torch.sum(y_true * y_pred, dim=-1)
    #     norm_true = torch.norm(y_true, dim=-1)
    #     norm_pred = torch.norm(y_pred, dim=-1)
    #     return torch.mean(torch.acos(torch.clamp(dot_product / (norm_true * norm_pred), -1.0, 1.0)))
    
    # # Compute SAM loss
    # sam = sam_loss(x.view(-1, x.shape[-1]), x_recon.view(-1, x_recon.shape[-1]))
    
    # # Pixel-wise Area Under Curve (AUC) comparison
    # def auc_loss(y_true, y_pred):
    #     # Compute AUC using trapezoidal rule for each pixel
    #     auc_true = torch.trapz(y_true, dim=-1)
    #     auc_pred = torch.trapz(y_pred, dim=-1)
        
    #     # Compute relative error for each pixel
    #     relative_error = torch.abs(auc_true - auc_pred) / (auc_true + 1e-8)
        
    #     # Return mean relative error across all pixels
    #     return torch.mean(relative_error)
    
    # # Compute AUC loss
    # auc_loss_value = auc_loss(x.squeeze(1), x_recon.squeeze(1))
    
    # ELBO
    elbo = -recon_loss + kl_weight * kl_loss
    
    # # Total loss (negative ELBO with additional SAM and AUC terms)
    # total_loss = -elbo + sam_weight * sam + auc_weight * auc_loss_value

    # return total_loss, -elbo, recon_loss, kl_loss, sam, auc_loss_value

    return recon_loss, kl_loss, elbo

def train_epoch(model, train_loader, optimizer, scheduler, config, epoch):
    """Train the model for one epoch."""
    model.train()
    # total_loss = 0 
    total_recon_loss = 0
    total_kl_loss = 0
    # total_sam_loss = 0  # New: track SAM loss
    total_elbo = 0 
    # total_auc_loss = 0
    device = torch.device(config['device'])
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        x = batch.to(device)
        optimizer.zero_grad()
        elbo, recon_loss, kl_loss = compute_loss(model, x, config['kl_weight'])
        # loss, elbo, recon_loss, kl_loss, sam_loss, auc_loss = compute_elbo_loss(model, x, kl_weight=config['kl_weight'], sam_weight=config['sam_weight'], auc_weight=config['auc_weight'])
        
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_value'])
        optimizer.step()
        
        # Accumulate losses
        batch_size = x.size(0)
        # total_loss += loss.item() * batch_size
        total_elbo += elbo.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size
        # total_sam_loss += sam_loss.item() * batch_size
        # total_auc_loss += auc_loss.item() * batch_size
        
        # Update progress bar
        progress_bar.set_postfix({
            # 'loss': loss.item(),
            'elbo': elbo.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            # 'sam': sam_loss.item(),
            # 'auc': auc_loss.item()
        })
        
        # Log batch-level metrics
        wandb.log({
            # "batch": epoch * len(train_loader) + batch_idx,
            # "total_loss": loss.item(),
            "elbo": elbo.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            # "sam_loss": sam_loss.item(),
            # 'auc_loss': auc_loss.item()
        })
        
    # Calculate average losses
    num_samples = len(train_loader.dataset)
    # avg_loss = total_loss / num_samples
    avg_elbo = total_elbo / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kl_loss = total_kl_loss / num_samples
    # avg_sam_loss = total_sam_loss / num_samples
    # avg_auc_loss = total_auc_loss / num_samples
    
    scheduler.step(avg_elbo)
    
    # return avg_loss, avg_elbo, avg_recon_loss, avg_kl_loss, avg_sam_loss, avg_auc_loss

    return avg_elbo, avg_recon_loss, avg_kl_loss

# def log_embeddings(model, data_loader, config, epoch):
#     """Log embeddings to wandb for visualization."""
#     model.eval()
#     device = torch.device(config['device'])
    
#     embeddings = []
#     original_data = []
    
#     with torch.no_grad():
#         for batch in data_loader:
#             x = batch.to(device)
#             mean, _ = model.encode(x)
#             embeddings.append(mean.cpu().numpy())
#             original_data.append(x.cpu().numpy())
    
#     embeddings = np.vstack(embeddings)
#     original_data = np.vstack(original_data)
    
#     # Create a wandb Table
#     columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
#     columns.append("original_data")
    
#     data = []
#     for emb, orig in tqdm(zip(embeddings, original_data), desc="Creating data for wandb table", total=len(embeddings)):
#         # Normalize the image data to [0, 1] range
#         img = orig.squeeze().sum(axis=-1)
#         if img.max() == img.min():
#             logger.warning("Image has no variation. Skipping this sample.")
#             continue
#         img = (img - img.min()) / (img.max() - img.min())
        
#         # Convert to uint8 and create wandb Image
#         img_uint8 = (img * 255).astype(np.uint8)
#         row = list(emb) + [wandb.Image(img_uint8)]
#         data.append(row)
    
#     table = wandb.Table(columns=columns, data=data)
    
#     # Log the table
#     wandb.log({f"embeddings_epoch_{epoch}": table})

def train_model(config):
    """Train the CVAE3D model."""
    logger.info("Starting model training")
    logger.debug(f"Configuration: {config}")

    # Initialize wandb
    wandb.init(project=config['wandb_project_name'], config=config)
    
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    # model = CVAE3D(input_shape=(24, 24, 480), latent_dim=config['latent_dim'], hidden_dims=config['hidden_dims']).to(device)
    model = CVAE3D(config).to(device)
    summary(model, input_size=(16, 1, 24, 24, 1312))
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['lr_patience'], factor=config['lr_factor'])

    train_loader = get_data_loader(config)
    logger.debug(f"Train loader batch size: {train_loader.batch_size}")
    logger.debug(f"Train loader length: {len(train_loader)}")

    best_loss = float('inf')
    stop_training = False

    for epoch in range(1, config['epochs'] + 1):
        if stop_training:
            break

        logger.info(f"Starting epoch {epoch}")
        # loss, elbo, recon_loss, kl_loss, sam_loss, auc_loss = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        elbo, recon_loss, kl_loss = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
                
        logger.info(f"Epoch {epoch}/{config['epochs']}, "
                    # f"Loss: {loss:.4f}, Recon Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}, ELBO: {elbo:.4f}, SAM: {sam_loss:.4f}, AUC: {auc_loss:.4f}")
                    f"Loss: {elbo:.4f}, Recon Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}")

        # # Log embeddings for visualization
        # if epoch % config['visualization_interval'] == 0:
        #     logger.info(f"Logging embeddings for epoch {epoch}")
        #     log_embeddings(model, train_loader, config, epoch) # CHANGED TO THE WANDB PROCESSING

        # Generate and log other visualizations every N epochs
        if epoch % config['visualization_interval'] == 0:
            logger.info(f"Generating additional visualizations for epoch {epoch}")
            model.eval()
            with torch.no_grad():
                try:
                    # Reconstruction visualization
                    logger.debug("Generating reconstruction visualization")
                    recon_fig = visualize_reconstructions(model, train_loader, device)
                    wandb.log({"reconstructions": wandb.Image(recon_fig)})

                except Exception as e:
                    logger.error(f"Error during visualization generation: {str(e)}")
                    logger.exception("Visualization error details:")

        # Save best model
        if elbo < best_loss and epoch % 20 == 0:
            best_loss = elbo
            logger.info(f"New best loss: {best_loss:.4f}. Saving model.")
            torch.save(model.state_dict(), f"{wandb.run.dir}/best_model.pth")

        # # Check for user input to stop training
        # if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        #     user_input = sys.stdin.readline().strip().lower()
        #     if user_input == 'x':
        #         logger.info("Training stopped by user. Saving model...")
        #         stop_training = True

    # Save final model
    logger.info("Training completed. Saving final model.")
    torch.save(model.state_dict(), f"{wandb.run.dir}/final_model.pth")

    wandb.finish()
    return model

if __name__ == "__main__":
    from anomaly_detection.config.config_handler import get_config
    
    logging.basicConfig(level=logging.INFO)
    
    config_path = 'path/to/your/config.yaml'
    config = get_config(config_path)
    
    trained_model = train_model(config)