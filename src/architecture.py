"""Classical and INN Autoencoder architectures.

Classes:
    MNISTAutoencoder: Classical Autoencoder architecture for MNIST.
    MNISTAutoencoder1024: MNIST Classical Autoencoder (increase hidden size 1024).
    MNISTAutoencoderDeep1024: MNIST Classical Autoencoder (hidden size 1024 + more layers).
    MNISTAutoencoder2048: MNIST Classical Autoencoder (increase hidden size 2048).
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------- #
#                              MNIST architectures                             #
# ---------------------------------------------------------------------------- #


class MNISTAutoencoder(nn.Module):
    """Classical Autoencoder architecture for MNIST dataset.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.

    Methods:
        forward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the MNIST classical autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor.
        """
        lat = self.encoder(inp)
        out = self.decoder(lat)
        out = out.view(-1, 1, 28, 28)
        return out


class MNISTAutoencoder1024(nn.Module):
    """MNIST Classical Autoencoder architecture with increased hidden size to 1024.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.

    Methods:
        forward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the MNIST classical autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor.
        """
        lat = self.encoder(inp)
        out = self.decoder(lat)
        out = out.view(-1, 1, 28, 28)
        return out


class MNISTAutoencoderDeep1024(nn.Module):
    """MNIST Classical Autoencoder architecture with hidden size 1024 and more layers.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.

    Methods:
        forward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the MNIST classical autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor.
        """
        lat = self.encoder(inp)
        out = self.decoder(lat)
        out = out.view(-1, 1, 28, 28)
        return out


class MNISTAutoencoder2048(nn.Module):
    """MNIST Classical Autoencoder architecture with increased hidden size 2048.

    Attributes:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.

    Methods:
        forward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the MNIST classical autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor.
        """
        lat = self.encoder(inp)
        out = self.decoder(lat)
        out = out.view(-1, 1, 28, 28)
        return out
