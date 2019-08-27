
from .gan_trainer import GAN_Trainer


class MakeupNetTrainer(GAN_Trainer):
    """The trainer for MakeupNet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)