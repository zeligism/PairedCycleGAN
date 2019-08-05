
import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeupNet(nn.Module):
    """The main module of the MakeupNet"""

    def __init__(self, name="makeupnet",
        num_channels=3, num_features=64, num_latent=100,
        depth=5, gan_type="gan", with_landmarks=False):
        """
        Initializes MakeupNet.

        Args:
            num_channels: the number of channels in the input images.
            num_features: controls the numbers of filters in each conv/up-conv layer.
            num_latent: the number of latent factors.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_latent = num_latent
        self.with_landmarks = with_landmarks

        self.G = Generator(num_channels, num_features, num_latent, with_landmarks)
        self.D = Discriminator(num_channels, num_features, num_latent, with_landmarks)
        self.weights_init()


    def weights_init(self, conv_sd=0.02, batchnorm_sd=0.02):
        """
        A method that initializes weights of `self` in-place.

        Args:
            conv_sd: the standard deviation of the conv/up-conv layers.
            batchnorm_sd: the standard deviation of the batch-norm layers.
        """
        def weights_init_apply(module):
            classname = module.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(module.weight.data, 0.0, conv_sd)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, batchnorm_sd)
                nn.init.constant_(module.bias.data, 0)

        self.apply(weights_init_apply)



class Generator(nn.Module):
    """The generator of the MakeupNet"""

    def __init__(self, num_channels, num_features, num_latent,
        depth=5, gan_type="gan", with_landmarks=False):
        super().__init__()

        modules = []

        # Layer #1 (in): Generating from latent vector
        modules.append(nn.ConvTranspose2d(num_latent, num_features * 8, 4, stride=1, padding=0, bias=False))
        modules.append(nn.BatchNorm2d(num_features * 8))
        modules.append(nn.ReLU(True))

        # Layer #2: state size. (num_features*8) x 4 x 4
        modules.append(nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(num_features * 4))
        modules.append(nn.ReLU(True))

        # Layer #3: state size. (num_features*4) x 8 x 8
        modules.append(nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(num_features * 2))
        modules.append(nn.ReLU(True))

        # Layer #4: state size. (num_features*2) x 16 x 16
        modules.append(nn.ConvTranspose2d(num_features * 2, num_features, 4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(num_features))
        modules.append(nn.ReLU(True))

        # Layer #5 (out): state size. (num_features) x 32 x 32
        modules.append(nn.ConvTranspose2d(num_features, num_channels, 4, stride=2, padding=1, bias=False))
        modules.append(nn.Tanh())

        self.main = nn.Sequential(*modules)


    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
    """The discriminator of the MakeupNet"""

    def __init__(self, num_channels, num_features, num_latent,
        depth=5, gan_type="gan", with_landmarks=False):
        super().__init__()

        modules = []

        # Layer #1 (in): input is (nc) x 64 x 64
        modules.append(nn.Conv2d(num_channels, num_features, 4, stride=2, padding=1, bias=False))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer #2: state size. (num_features) x 32 x 32
        modules.append(nn.Conv2d(num_features, num_features * 2, 4, stride=2, padding=1, bias=False))
        if gan_type != "wgan-gp": modules.append(nn.BatchNorm2d(num_features * 2))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer #3: state size. (num_features*2) x 16 x 16
        modules.append(nn.Conv2d(num_features * 2, num_features * 4, 4, stride=2, padding=1, bias=False))
        if gan_type != "wgan-gp": modules.append(nn.BatchNorm2d(num_features * 4))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer #4: state size. (num_features*4) x 8 x 8
        modules.append(nn.Conv2d(num_features * 4, num_features * 8, 4, stride=2, padding=1, bias=False))
        if gan_type != "wgan-gp": modules.append(nn.BatchNorm2d(num_features * 8))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # Layer #5 (out): state size. (num_features*8) x 4 x 4
        modules.append(nn.Conv2d(num_features * 8, 1, 4, stride=1, padding=0, bias=False))
        modules.append(nn.Sigmoid())

        self.main = nn.Sequential(*modules)


    def forward(self, inputs):
        return self.main(inputs).view(-1)


