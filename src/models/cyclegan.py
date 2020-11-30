
import torch.nn as nn

from .dcgan import DCGAN
from .maskgan import MaskGAN


class CycleGAN(nn.Module):
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

        self.applier = DCGAN(**model_config)
        self.remover = DCGAN(**model_config)


class MaskCycleGAN(nn.Module):
    def __init__(self,
                 num_features=64,
                 image_channels=3,
                 image_size=64,
                 gan_type="gan",
                 with_reference=False,
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
            "with_reference": with_reference,
        }

        self.applier = MaskGAN(**model_config)
        self.remover = MaskGAN(**model_config)

