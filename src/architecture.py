"""Classical and INN Autoencoder architectures.

Functions:
    get_mnist_inn_autoencoder (None) -> RGNet: Returns the MNIST INN autoencoder architecture.
    get_cifar10_inn_autoencoder (None) -> RGNet: Returns the CIFAR10 INN autoencoder architecture.
    get_celeba_inn_autoencoder (None) -> RGNet: Returns the CelebA INN autoencoder architecture.

Classes:
    MNISTAutoencoder: Classical Autoencoder architecture for MNIST.
    MNISTAutoencoder1024: MNIST Classical Autoencoder (increase hidden size 1024).
    MNISTAutoencoderDeep1024: MNIST Classical Autoencoder (hidden size 1024 + more layers).
    MNISTAutoencoder2048: MNIST Classical Autoencoder (increase hidden size 2048).
    CIFAR10Autoencoder: Classical Autoencoder architecture for CIFAR10.
    CelebAAutoencoder: Classical Autoencoder architecture for CelebA.
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
    """Return the INN autoencoder architecture for MNIST.

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


# ---------------------------------------------------------------------------- #
#                            CIFAR 10 architectures                            #
# ---------------------------------------------------------------------------- #


class CIFAR10Autoencoder(nn.Module):
    """Classical Autoencoder architecture for CIFAR10 dataset.

    Attributes:
        encoder (nn.Module): Encoder network.
        bottleneck (nn.Module): Bottleneck fully-connected layer.
        decoder (nn.Module): Decoder network.

    Methods:
        forward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the CIFAR10 autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            # output: (64, 30, 30)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # output: (128, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # output: (128, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # output: (256, 12, 12)
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # output: (512, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # output: (512, 5, 5)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            # output: (1024, 3, 3)
            nn.ReLU(True),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(1024 * 3 * 3, bottleneck),
            nn.Linear(bottleneck, 1024 * 3 * 3),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=0),
            # output: (512, 5, 5)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            # output: (512, 10, 10)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0),
            # output: (256, 12, 12)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            # output: (128, 14, 14)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            # output: (128, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),
            # output: (64, 30, 30)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=0),
            # output: (3, 32, 32)
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
        lat_flat = lat.view(-1, 1024 * 3 * 3)
        lat_flat = self.bottleneck(lat_flat)
        lat = lat.view(-1, 1024, 3, 3)
        out = self.decoder(lat)
        return out


def get_cifar10_inn_autoencoder() -> ReversibleGraphNet:
    """Return the INN autoencoder architecture for CIFAR10 dataset.

    Returns:
        coder (ReversibleGraphNet): CIFAR10 INN autoencoder architecture.
    """
    img_dims = [3, 32, 32]

    input_node = fr.InputNode(*img_dims, name="input")

    reshape_node_1 = fr.Node([(input_node, 0)], re.haar_multiplex_layer, {}, name="r1")

    conv_node_1 = fr.Node(
        [(reshape_node_1, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv1",
    )

    conv_node_2 = fr.Node(
        [(conv_node_1, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv2",
    )

    conv_node_3 = fr.Node(
        [(conv_node_2, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
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
        {
            "F_class": fu.F_small_connected,
            "F_args": {"internal_size": 1000},
            "clamp": 1,
        },
        name="fc",
    )

    reshape_node_3 = fr.Node(
        [(fully_con_node, 0)], re.reshape_layer, {"target_dim": (12, 16, 16)}, name="r3"
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
        fully_con_node,
        reshape_node_1,
        reshape_node_2,
        reshape_node_3,
        reshape_node_4,
    ]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


# ---------------------------------------------------------------------------- #
#                             CelebA architectures                             #
# ---------------------------------------------------------------------------- #


class CelebAAutoencoder(nn.Module):
    """Classical Autoencoder architecture for CelebA dataset.

    Attributes:
        encoder (nn.Module): Encoder network.
        bottleneck (nn.Module): Bottleneck fully-connected layer.
        decoder (nn.Module): Decoder network.

    Methods:
        foward (torch.Tensor): Forward pass of the network.
    """

    def __init__(self, bottleneck: int) -> None:
        """Initialize the CelebA autoencoder architecture.

        Args:
            bottleneck (int): Size of bottleneck layer.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=0),
            # output: (128, 108, 88)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # output: (256, 54, 44)
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=(0, 1)),
            # output: (512, 26, 22)
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=0),
            # output: (1024, 12, 10)
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=0),
            # output: (1024, 5, 4)
            nn.ReLU(True),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(1024 * 5 * 4, bottleneck),
            nn.Linear(bottleneck, 1024 * 5 * 4),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=0),
            # output: (1024, 12, 10)
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=0),
            # output: (512, 26, 22)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=(0, 1)),
            # output: (256, 54, 44)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # output: (128, 108, 88)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=0),
            # output: (3, 218, 178)
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
        lat_flat = lat.view(-1, 1024 * 5 * 4)
        lat_flat = self.bottleneck(lat_flat)
        lat = lat.view(-1, 1024, 5, 4)
        out = self.decoder(lat)
        return out


def get_celeba_inn_autoencoder() -> ReversibleGraphNet:
    """Return the INN autoencoder architecture for CelebA dataset.

    Returns:
        coder (ReversibleGraphNet): CelebA INN autoencoder architecture.
    """
    # pylint: disable=too-many-locals
    # FrEIA framework forces INN architecture construction with local variables
    img_dims = [3, 128, 128]

    input_node = fr.InputNode(*img_dims, name="input")

    reshape_node_1 = fr.Node([(input_node, 0)], re.haar_multiplex_layer, {}, name="r1")

    conv_node_11 = fr.Node(
        [(reshape_node_1, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv11",
    )

    conv_node_12 = fr.Node(
        [(conv_node_11, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv12",
    )

    conv_node_13 = fr.Node(
        [(conv_node_12, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv13",
    )

    reshape_node_2 = fr.Node(
        [(conv_node_13, 0)], re.haar_multiplex_layer, {}, name="r2"
    )

    conv_node_21 = fr.Node(
        [(reshape_node_2, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv21",
    )

    conv_node_22 = fr.Node(
        [(conv_node_21, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv22",
    )

    conv_node_23 = fr.Node(
        [(conv_node_22, 0)],
        la.glow_coupling_layer,
        {"F_class": fu.F_conv, "F_args": {"channels_hidden": 128}, "clamp": 1},
        name="conv23",
    )

    reshape_node_3 = fr.Node(
        [(conv_node_23, 0)],
        re.reshape_layer,
        {
            "target_dim": (
                (img_dims[0] * 4 * 4)
                * int(img_dims[1] / 2 / 2)
                * int(img_dims[2] / 2 / 2),
            )
        },
        name="r3",
    )

    fully_con_node = fr.Node(
        [(reshape_node_3, 0)],
        la.rev_multiplicative_layer,
        {"F_class": fu.F_small_connected, "F_args": {"internal_size": 200}, "clamp": 1},
        name="fc",
    )

    reshape_node_4 = fr.Node(
        [(fully_con_node, 0)],
        re.reshape_layer,
        {
            "target_dim": (
                img_dims[0] * 4 * 4,
                int(img_dims[1] / 2 / 2),
                int(img_dims[2] / 2 / 2),
            )
        },
        name="r4",
    )

    reshape_node_5 = fr.Node(
        [(reshape_node_4, 0)], re.haar_restore_layer, {}, name="r5"
    )

    reshape_node_6 = fr.Node(
        [(reshape_node_5, 0)], re.haar_restore_layer, {}, name="r6"
    )

    output_node = fr.OutputNode([(reshape_node_6, 0)], name="output")

    nodes = [
        input_node,
        output_node,
        conv_node_11,
        conv_node_12,
        conv_node_13,
        conv_node_21,
        conv_node_22,
        conv_node_23,
        fully_con_node,
        reshape_node_1,
        reshape_node_2,
        reshape_node_3,
        reshape_node_4,
        reshape_node_5,
        reshape_node_6,
    ]

    coder = fr.ReversibleGraphNet(nodes, 0, 1)

    return coder


# ---------------------------------------------------------------------------- #
#                                  Entry Point                                 #
# ---------------------------------------------------------------------------- #

MODEL_ARCHITECTURES = {
    "mnist_classic": MNISTAutoencoder,
    "mnist_classic1024": MNISTAutoencoder1024,
    "mnist_classicDeep1024": MNISTAutoencoderDeep1024,
    "mnist_classic2048": MNISTAutoencoder2048,
    "mnist_inn": get_mnist_inn_autoencoder,
    "cifar_classic": CIFAR10Autoencoder,
    "cifar_inn": get_cifar10_inn_autoencoder,
    "celeba_clasic": CelebAAutoencoder,
    "celeba_inn": get_celeba_inn_autoencoder,
}
