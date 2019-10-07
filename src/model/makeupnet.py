
import torch
import torch.nn as nn

from .gan import DCGAN, DCGAN_Discriminator
from .residual import ResidualBlock


class _MakeupNet(DCGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MakeupNet(nn.Module):
    def __init__(self, num_features, image_channels=3, image_size=64, gan_type="gan", **kwargs):
        super().__init__()

        self.num_features = num_features
        self.image_channels = image_channels
        self.image_size = image_size
        self.gan_type = gan_type

        makeupnet_params = {
            "image_channels": image_channels,
            "num_features": num_features,
            "image_size": image_size,
            "gan_type": gan_type,
        }

        self.remover = MakeupMaskGAN(**makeupnet_params)
        self.applier = MakeupMaskGAN(**makeupnet_params, with_reference=True)
        self.style_D = StyleDiscriminator(**makeupnet_params)


class MakeupMaskGAN(nn.Module):
    def __init__(self, num_features,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 with_reference=False):
        super().__init__()

        D_params = {
            "num_features": num_features,
            "image_channels": image_channels,
            "image_size": image_size,
            "gan_type": gan_type,
        }
        G_params = {
            "num_features": num_features,
            "with_reference": with_reference,
        }

        self.D = DCGAN_Discriminator(**D_params)
        self.G = MakeupMaskGenerator(**G_params)


class MakeupMaskGenerator(nn.Module):
    """A neural network that generates a mask that applies or removes makeup."""
    def __init__(self, num_features=64, with_reference=False):
        super().__init__()

        self.num_features = num_features
        self.with_reference = with_reference

        def make_features_extractor():
            return nn.Sequential(
                nn.Conv2d(3, self.num_features, 7, padding=3, bias=False),
                nn.ReLU(),
            )

        # Extract features from source
        self.source_features_extractor = make_features_extractor()

        # Extract features from reference
        if self.with_reference:
            self.reference_features_extractor = make_features_extractor()


        # Double the number of features in the mask generator if with reference
        if self.with_reference:
            num_features *= 2

        self.mask_generator = nn.Sequential(
            ResidualBlock(num_features, num_features),
            ResidualBlock(num_features, num_features, dilation=(2,2)),
            ResidualBlock(num_features, num_features, dilation=(4,4)),
            nn.Conv2d(num_features, num_features, 3, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_features, 3, 3, padding=1, bias=False),
            nn.Tanh(),
        )


    def forward(self, source, reference=None):

        assert reference is None or self.with_reference

        features = self.source_features_extractor(source)

        if self.with_reference:
            reference_features = self.reference_features_extractor(reference)
            features = torch.cat([features, reference_features], dim=1)

        mask = self.mask_generator(features)

        return source + mask


class StyleDiscriminator(DCGAN_Discriminator):
    def __init__(self, *args, **kwargs):

        kwargs["image_channels"] *= 2  # @XXX: extract features from img first?
        super().__init__(*args, **kwargs)


