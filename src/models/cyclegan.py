
import torch.nn as nn

from .dcgan import DCGAN
from .maskgan import MaskGAN


class CycleGAN(nn.Module):
    def __init__(self, num_features,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 gan_class=DCGAN,
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

        self.applier = gan_class(**model_config)
        self.remover = gan_class(**model_config)


class MaskCycleGAN(CycleGAN):
    def __init__(self, *args, **kwargs):
        kwargs["gan_class"] = MaskGAN
        super().__init__(*args, **kwargs)

