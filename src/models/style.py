
import torch
import torch.nn as nn

EPS = 1e-8

# @TODO: Style modules.

class LatentMapper(nn.Module):
    """
    Latent mapper module.
    """

    def __init__(self, latent_dim, layer_dim, interlatent_dim, num_layers):
        super().__init__()
        
        dims = [latent_dim] + num_layers * [layer_dim] + [interlatent_dim]
        self.main = nn.Sequential(nn.Linear(in_dim, out_dim)
                                  for in_dim, out_dim in zip(dims, dims[1:]))


    def forward(self, z):
        return self.main(z)


class ChannelNoise(nn.Module):
    """
    Channel noise injection module.
    Adds a linearly transformed noise to a convolution layer.
    """

    def __init__(self, num_channels, std=0.02):
        super().__init__()
        self.std = std
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))


    def forward(self, x):
        noise_size = [x.size()[0], 1, *x.size()[2:]]  # single channel
        noise = self.std * torch.randn(noise_size).to(x)

        return x + self.scale * noise


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.linear = nn.Linear(2*num_channels, 2*num_channels)


    def forward(self, x, transformed_latent):
        
        # Unpack dims of x
        batch_size, num_channels, height, width = x.size()

        # Group height and width dims and get their mean and std
        x = x.view(batch_size, num_channels, -1)
        x_mean = x.mean(dim=2)
        x_std  = x.std(dim=2)

        # Regroup x's dims back and calculate the normalized x
        x = x.view([batch_size, num_channels, height, width])
        normalized_x = (x - x_mean) / x_std

        # Transform intermediate latent representation to style
        style = self.linear(transformed_latent)
        # Unpack scale and bias
        style_scale = style[:num_channels].view(1, num_channels, 1, 1)
        style_bias  = style[num_channels:].view(1, num_channels, 1, 1)

        return style_scale * normalized_x + style_bias


class PixelNorm(nn.Module):
    """
    Pixel norm.
    """

    def __init__(self):
        super().__init__()


    def forward(self, x):
        return pixel_norm(x)


def pixel_norm(x):
    num_channels = x.size()[1]
    pixel_mean = x.sum(dim=1) / num_channels
    return x / (pixel_mean.sqrt() + EPS)




