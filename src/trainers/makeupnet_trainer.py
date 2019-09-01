
import torch

from .gan_trainer import GAN_Trainer


class MakeupNetTrainer(GAN_Trainer):
    """The trainer for MakeupNet."""

    def __init__(self, model, dataset,
        **kwargs):
        """
        Initializes GAN_Trainer.

        Args:
            model: The makeup net.
            dataset: The makeup dataset.
        """

        super().__init__(model, dataset, **kwargs)