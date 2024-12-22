# anomaly_detection/models/cvae3d.py
# Author: Seyfal Sultanov 

import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE3D(nn.Module):
    """
    Convolutional Variational Autoencoder for 3D EELS data.

    This model is designed to work with input data of shape (batch_size, 1, 24, 24, 240),
    representing EELS spectra.

    Attributes:
        latent_dim (int): Dimension of the latent space.
        encoder (nn.Sequential): Encoder network.
        decoder (nn.Sequential): Decoder network.
    """

    def __init__(self, config):
        """
        Initialize the CVAE3D model.

        Args:
            latent_dim (int): Dimension of the latent space.
        """
        super(CVAE3D, self).__init__()
        self.latent_dim = config['latent_dim']
        
        # ==========================================================================================
        # Layer (type:depth-idx)                   Output Shape              Param #
        # ==========================================================================================
        # CVAE3D                                   [63, 1, 24, 24, 480]      --
        # ├─Sequential: 1-1                        [63, 48]                  --
        # │    └─Conv3d: 2-1                       [63, 32, 24, 24, 240]     256
        # │    └─ReLU: 2-2                         [63, 32, 24, 24, 240]     --
        # │    └─Conv3d: 2-3                       [63, 64, 24, 24, 120]     92,224
        # │    └─ReLU: 2-4                         [63, 64, 24, 24, 120]     --
        # │    └─Conv3d: 2-5                       [63, 128, 12, 12, 60]     221,312
        # │    └─ReLU: 2-6                         [63, 128, 12, 12, 60]     --
        # │    └─Conv3d: 2-7                       [63, 256, 6, 6, 30]       884,992
        # │    └─ReLU: 2-8                         [63, 256, 6, 6, 30]       --
        # │    └─Flatten: 2-9                      [63, 276480]              --
        # │    └─Linear: 2-10                      [63, 512]                 141,558,272
        # │    └─ReLU: 2-11                        [63, 512]                 --
        # │    └─Linear: 2-12                      [63, 48]                  24,624
        # ├─Sequential: 1-2                        [63, 1, 24, 24, 480]      --
        # │    └─Linear: 2-13                      [63, 512]                 12,800
        # │    └─ReLU: 2-14                        [63, 512]                 --
        # │    └─Linear: 2-15                      [63, 276480]              141,834,240
        # │    └─ReLU: 2-16                        [63, 276480]              --
        # │    └─Unflatten: 2-17                   [63, 256, 6, 6, 30]       --
        # │    └─ConvTranspose3d: 2-18             [63, 128, 12, 12, 60]     884,864
        # │    └─ReLU: 2-19                        [63, 128, 12, 12, 60]     --
        # │    └─ConvTranspose3d: 2-20             [63, 64, 24, 24, 120]     221,248
        # │    └─ReLU: 2-21                        [63, 64, 24, 24, 120]     --
        # │    └─ConvTranspose3d: 2-22             [63, 32, 24, 24, 240]     92,192
        # │    └─ReLU: 2-23                        [63, 32, 24, 24, 240]     --
        # │    └─ConvTranspose3d: 2-24             [63, 1, 24, 24, 480]      225
        # ==========================================================================================
        # Total params: 285,827,249
        # Trainable params: 285,827,249
        # Non-trainable params: 0
        # Total mult-adds (Units.TERABYTES): 2.85
        # ==========================================================================================
        # Input size (MB): 69.67
        # Forward/backward pass size (MB): 10451.48
        # Params size (MB): 1143.31
        # Estimated Total Size (MB): 11664.47
        # ==========================================================================================
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1,1,7), stride=(1,1,2), padding=(0,0,3)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,5), stride=(1,1,2), padding=(1,1,2)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6 * 82, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim * 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 6 * 6 * 82),
            nn.ReLU(),
            nn.Unflatten(1, (256, 6, 6, 82)),
            nn.ConvTranspose3d(256, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3,3,5), stride=(1,1,2), padding=(1,1,2), output_padding=(0,0,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=(1,1,7), stride=(1,1,2), padding=(0,0,3), output_padding=(0,0,1))
        )

    def encode(self, x):
        """
        Encode the input data into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 24, 24, 240).

        Returns:
            tuple: Mean and log variance of the latent space distribution.
        """
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Perform the reparameterization trick.

        Args:
            mean (torch.Tensor): Mean of the latent space distribution.
            logvar (torch.Tensor): Log variance of the latent space distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, apply_sigmoid=False):
        """
        Decode the latent vector into the output space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        return logits

    def forward(self, x):
        """
        Forward pass through the CVAE3D model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 24, 24, 240).

        Returns:
            tuple: Reconstructed output, mean, and log variance of the latent space distribution.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar