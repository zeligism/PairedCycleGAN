
import torch.nn as nn

from .gan import Discriminator, Generator


class MakeupNet(nn.Module):
    """The main module of the MakeupNet"""

    def __init__(self,
        name="MakeupNet",
        gan_type="gan",
        num_channels=3,
        num_features=64,
        num_latents=128,
        depth=5,
        conv_std=0.02,
        batchnorm_std=0.02,
        with_landmarks=False,):
        """
        Initializes MakeupNet. @TODO

        Args:
            num_channels: the number of channels in the input images.
            num_features: controls the numbers of filters in each conv/up-conv layer.
            num_latents: the number of latent factors.
        """
        super().__init__()

        self.name = name
        self.gan_type = gan_type
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_latents = num_latents
        self.depth = depth
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

        self.D = Discriminator(**D_params)
        self.G = Generator(**G_params)

        self.weights_init(conv_std=0.02, batchnorm_std=0.02)


    def weights_init(self, conv_std=0.02, batchnorm_std=0.02):
        """
        A method that initializes weights of `self` in-place.

        Args:
            conv_std: the standard deviation of the conv/up-conv layers.
            batchnorm_std: the standard deviation of the batch-norm layers.
        """
        def weights_init_apply(module):
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, 0.0, conv_std)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, batchnorm_std)
                nn.init.constant_(module.bias.data, 0)

        self.apply(weights_init_apply)

