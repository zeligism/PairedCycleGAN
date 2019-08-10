
import torch.nn as nn


class Discriminator(nn.Module):
    """The discriminator of the MakeupNet"""

    def __init__(self, gan_type, num_channels, num_features, depth=5):
        super().__init__()

        using_gradient_penalty = gan_type == "wgan-gp"
        use_batchnorm = not using_gradient_penalty

        # input is num_channels x H x W
        self.main = nn.Sequential(
            DiscriminatorBlock(num_channels, num_features),
            DiscriminatorBlock(num_features * 1, num_features * 2, use_batchnorm=use_batchnorm),
            DiscriminatorBlock(num_features * 2, num_features * 4, use_batchnorm=use_batchnorm),
            DiscriminatorBlock(num_features * 4, num_features * 8, use_batchnorm=use_batchnorm),
            nn.Conv2d(num_features * 8, 1,
                      kernel_size=4, stride=1, padding=0, bias=False),
        )

        if gan_type == "gan":
            self.main.add_module("D-Sigmoid", nn.Sigmoid())


    def forward(self, inputs):
        return self.main(inputs).view(-1)


class DiscriminatorBlock(nn.Module):
    """
    A discriminator convolutional block.
    Default stride and padding half the size of features,
    e.g. if input is [in_channels, 64, 64], output will be [out_channels, 32, 32].
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=1, use_batchnorm=False):
        super().__init__()

        modules = []

        modules.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False))
        if use_batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.main = nn.Sequential(*modules)


    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """The generator of the MakeupNet"""

    def __init__(self, gan_type, num_latents, num_features, num_channels, depth=5):
        super().__init__()

        # @XXX: is gan_type useless here?

        self.main = nn.Sequential(
            GeneratorBlock(num_latents, num_features * 8, stride=1, padding=0),
            GeneratorBlock(num_features * 8, num_features * 4),
            GeneratorBlock(num_features * 4, num_features * 2),
            GeneratorBlock(num_features * 2, num_features * 1),
            nn.ConvTranspose2d(num_features, num_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )


    def forward(self, inputs):
        inputs = inputs.view(*inputs.size(), 1, 1)  # add H and W dimensions
        return self.main(inputs)


class GeneratorBlock(nn.Module):
    """
    A generator convolutional block.
    Default stride and padding double the size of features,
    e.g. if input is [in_channels, 32, 32], output will be [out_channels, 64, 64].
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                               stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        return self.main(x)

