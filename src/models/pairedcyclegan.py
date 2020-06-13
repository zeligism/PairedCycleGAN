
import torch.nn as nn

from .dcgan import DCGAN_Discriminator
from .maskgan import MaskGAN


class PairedCycleGAN(nn.Module):
    def __init__(self,
                 num_features=64,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 **kwargs):
        super().__init__()

        self.num_features = num_features
        self.image_channels = image_channels
        self.image_size = image_size
        self.gan_type = gan_type

        model_config = {
            "image_channels": image_channels,
            "num_features": num_features,
            "image_size": image_size,
            "gan_type": gan_type,
        }

        self.remover = MaskGAN(**model_config)
        self.applier = MaskGAN(**model_config, with_reference=True)
        self.style_D = StyleDiscriminator(**model_config)


class StyleDiscriminator(DCGAN_Discriminator):
    def __init__(self, *args, **kwargs):

        kwargs["image_channels"] *= 2  # XXX: extract features from img first?
        super().__init__(*args, **kwargs)


