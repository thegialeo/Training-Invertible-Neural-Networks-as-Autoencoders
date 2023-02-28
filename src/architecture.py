"""Classical and INN Autoencoder architectures.

Classes:
    MNISTAutoencoder: Classical Autoencoder architecture for MNIST.
    MNISTAutoencoder1024: MNIST Classical Autoencoder (increase hidden size 1024).
    MNISTAutoencoderDeep1024: MNIST Classical Autoencoder (hidden size 1024 + more layers).
    MNISTAutoencoder2048: MNIST Classical Autoencoder (increase hidden size 2048).
"""

import torch
from torch import nn

from FrEIA import framework as fr
from FrEIA.framework import ReversibleGraphNet
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re

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


def get_mnist_inn_autoencoder() -> ReversibleGraphNet:
    """Get the MNIST INN autoencoder architecture.

    Returns:
        coder (ReversibleGraphNet): MNIST INN autoencoder architecture.
    """
    img_dims = [1, 28, 28]

    input_node = fr.InputNode(*img_dims, name="input")

    reshape_node_1 = fr.Node([(input_node, 0)], re.haar_multiplex_layer, {}, name="r1")

    conv_node_1 = fr.Node(
        [(reshape_node_1, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 100}, "clamp": 1},
        name="conv1",
    )

    conv_node_2 = fr.Node(
        [(conv_node_1, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 100}, "clamp": 1},
        name="conv2",
    )

    conv_node_3 = fr.Node(
        [(conv_node_2, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 100}, "clamp": 1},
        name="conv3",
    )

    reshape_node_2 = fr.Node(
        [(conv_node_3, 0)],
        re.reshape_layer,
        {"target_dim": (img_dims[0] * img_dims[1] * img_dims[2],)},
        name="r2",
    )

    fully_con_node = fr.Node(
        [(reshape_node_2, 0)],
        la.rev_multiplicative_layer,
        {"F_class": fu.F_small_connected, "F_args": {"internal_size": 180}, "clamp": 1},
        name="fc",
    )

    reshape_node_3 = fr.Node(
        [(fully_con_node, 0)], re.reshape_layer, {"target_dim": (4, 14, 14)}, name="r3"
    )

    reshape_node_4 = fr.Node(
        [(reshape_node_3, 0)], re.haar_restore_layer, {}, name="r4"
    )

    output_node = fr.OutputNode([(reshape_node_4, 0)], name="output")

    nodes = [
        input_node,
        output_node,
        conv_node_1,
        conv_node_2,
        conv_node_3,
        reshape_node_1,
        reshape_node_2,
        reshape_node_3,
        reshape_node_4,
        fully_con_node,
    ]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder
