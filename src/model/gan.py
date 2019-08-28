
import torch.nn as nn

from .init_utils import create_weights_init


class DCGAN(nn.Module):
    """Deep Convolutional Generative Adversarial Network"""

    def __init__(self,
        gan_type="gan",
        num_channels=3,
        num_features=64,
        num_latents=128,
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
        self.conv_std = conv_std
        self.batchnorm_std = batchnorm_std
        self.with_landmarks = with_landmarks

        D_params = {
            "gan_type": gan_type,
            "num_channels": num_channels,
            "num_features": num_features,
        }
        G_params = {
            "gan_type": gan_type,
            "num_latents": num_latents,
            "num_features": num_features,
            "num_channels": num_channels,
        }

        self.D = DCGAN_Discriminator(**D_params)
        self.G = DCGAN_Generator(**G_params)

        weights_init = create_weights_init()
        self.apply(weights_init)


class DCGAN_Discriminator(nn.Module):
    """The discriminator of the MakeupNet"""

    def __init__(self, gan_type, num_channels, num_features):
        super().__init__()

        using_gradient_penalty = gan_type == "wgan-gp"
        use_batchnorm = not using_gradient_penalty

        # input is num_channels x H x W
        self.main = nn.Sequential(
            DCGAN_DiscriminatorBlock(num_channels, num_features),
            DCGAN_DiscriminatorBlock(num_features * 1, num_features * 2, use_batchnorm=use_batchnorm),
            DCGAN_DiscriminatorBlock(num_features * 2, num_features * 4, use_batchnorm=use_batchnorm),
            DCGAN_DiscriminatorBlock(num_features * 4, num_features * 8, use_batchnorm=use_batchnorm),
            nn.Conv2d(num_features * 8, 1, 4, stride=1, padding=0, bias=False),
        )

        if gan_type == "gan":
            self.main.add_module("D-Sigmoid", nn.Sigmoid())


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

    def __init__(self, gan_type, num_latents, num_features, num_channels):
        super().__init__()

        # @XXX: is gan_type useless here?

        self.main = nn.Sequential(
            DCGAN_GeneratorBlock(num_latents, num_features * 8, stride=1, padding=0),
            DCGAN_GeneratorBlock(num_features * 8, num_features * 4),
            DCGAN_GeneratorBlock(num_features * 4, num_features * 2),
            DCGAN_GeneratorBlock(num_features * 2, num_features * 1),
            nn.ConvTranspose2d(num_features, num_channels, 4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        )


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

