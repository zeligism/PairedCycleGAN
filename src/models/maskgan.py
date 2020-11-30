
import torch
import torch.nn as nn

from .dcgan import DCGAN_Discriminator
from .residual import ResidualBlock


class MaskGAN(nn.Module):
    def __init__(self,
                 num_features=64,
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
            "num_features": 3*num_features,  # XXX: due to parameters inbalance
            "with_reference": with_reference,
        }

        self.D = DCGAN_Discriminator(**D_params)
        self.G = MaskGenerator(**G_params)


class MaskGenerator(nn.Module):
    """A neural network that generates a mask to apply."""
    def __init__(self, num_features=64, with_reference=False):
        super().__init__()

        self.num_features = num_features
        self.with_reference = with_reference

        def make_features_extractor(num_features):
            return nn.Sequential(
                nn.Conv2d(3, num_features, 7, padding=3, bias=False),
                nn.ReLU(),
            )

        # Extract features from source
        self.source_features_extractor = make_features_extractor(self.num_features)

        # Extract features from reference
        if self.with_reference:
            self.reference_features_extractor = make_features_extractor(self.num_features)


        # Double the number of features in the mask generator if with reference
        if self.with_reference:
            num_features *= 2

        self.mask_generator = nn.Sequential(
            ResidualBlock(num_features, num_features),
            ResidualBlock(num_features, num_features, dilation=(2,2)),
            ResidualBlock(num_features, num_features, dilation=(4,4)),
            ResidualBlock(num_features, num_features, dilation=(8,8)),
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

        return (source + mask).clamp(-1,1) # XXX: range could go outside [-1, 1] !!!

