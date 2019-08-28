
import torch.nn as nn

from math import log2
from .init_utils import create_weights_init


class DCGAN(nn.Module):
    """Deep Convolutional Generative Adversarial Network"""

    def __init__(self,
        gan_type="gan",
        num_channels=3,
        num_features=64,
        num_latents=128,
        image_size=64,
        conv_std=0.02,
        batchnorm_std=0.02,
        with_landmarks=False):
        """
        Initializes DCGAN.

        Args: @TODO
            num_channels: the number of channels in the input images.
            num_features: controls the numbers of filters in each conv/up-conv layer.
            num_latents: the number of latent factors.
        """
        super().__init__()

        self.gan_type = gan_type
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_latents = num_latents
        self.image_size = image_size
        self.conv_std = conv_std
        self.batchnorm_std = batchnorm_std
        self.with_landmarks = with_landmarks

        D_params = {
            "gan_type": gan_type,
            "num_channels": num_channels,
            "num_features": num_features,
            "image_size": image_size,
        }
        G_params = {
            "gan_type": gan_type,
            "num_latents": num_latents,
            "num_features": num_features,
            "num_channels": num_channels,
            "image_size": image_size,
        }

        self.D = DCGAN_Discriminator(**D_params)
        self.G = DCGAN_Generator(**G_params)

        weights_init = create_weights_init()
        self.apply(weights_init)


class DCGAN_Discriminator(nn.Module):
    """The discriminator of the MakeupNet"""

    def __init__(self, gan_type, num_channels, num_features, image_size, max_features=512):
        super().__init__()

        using_gradient_penalty = gan_type == "wgan-gp"
        use_batchnorm = not using_gradient_penalty

        # Count number of layers (including input) and calculate feature sizes
        num_layers = int(round(log2(image_size // 4)))
        features = [min(num_features * 2**i, max_features) for i in range(num_layers)]

        modules = []

        # Input layer
        modules += [DCGAN_DiscriminatorBlock(num_channels, features[0])]

        # Intermediate layers
        for in_features, out_features in zip(features, features[1:]):
            modules += [DCGAN_DiscriminatorBlock(in_features, out_features, use_batchnorm=use_batchnorm)]
        
        # Output layer (feature_size = 4 -> 1)
        modules += [nn.Conv2d(features[-1], 1, 4, bias=False)]

        if gan_type == "gan":
            modules += [nn.Sigmoid()]

        self.main = nn.Sequential(*modules)


    def forward(self, inputs):
        return self.main(inputs).view(-1)


class DCGAN_DiscriminatorBlock(nn.Module):
    """
    A discriminator convolutional block.
    Default stride and padding half the size of features,
    e.g. if input is [in_channels, 64, 64], output will be [out_channels, 32, 32].
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=1, use_batchnorm=False):
        super().__init__()

        modules = []

        modules += [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        modules += [nn.BatchNorm2d(out_channels)] if use_batchnorm else []
        modules += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*modules)


    def forward(self, x):
        return self.main(x)


class DCGAN_Generator(nn.Module):
    """The generator of the MakeupNet"""

    def __init__(self, gan_type, num_latents, num_features, num_channels, image_size, max_features=512):
        super().__init__()

        # @XXX: is gan_type useless here?

        # Count number of layers (including input) and calculate feature sizes
        num_layers = int(round(log2(image_size // 4)))
        features = [min(num_features * 2**i, max_features) for i in range(num_layers)][::-1]

        modules = []

        # Input layer
        modules += [DCGAN_GeneratorBlock(num_latents, features[0], stride=1, padding=0)]

        # Intermediate layers
        for in_features, out_features in zip(features, features[1:]):
            modules += [DCGAN_GeneratorBlock(in_features, out_features)]
        
        # Output layer
        modules += [nn.ConvTranspose2d(features[-1], num_channels, 4,
                                       stride=2, padding=1, bias=False)]

        modules += [nn.Tanh()]

        self.main = nn.Sequential(*modules)


    def forward(self, inputs):
        inputs = inputs.view(*inputs.size(), 1, 1)  # add H and W dimensions
        return self.main(inputs)


class DCGAN_GeneratorBlock(nn.Module):
    """
    A generator convolutional block.
    Default stride and padding double the size of features,
    e.g. if input is [in_channels, 32, 32], output will be [out_channels, 64, 64].
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4,
                               stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        return self.main(x)

