
import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .utils.init_utils import init_optim
from .utils.gan_utils import *


class CycleGAN_Trainer(BaseTrainer):
    """The trainer for CycleGAN."""

    def __init__(self, model, dataset,
                 D_optim_config={},
                 G_optim_config={},
                 D_iters=5,
                 clamp=(-0.01, 0.01),
                 gp_coeff=10.,
                 stats_interval=50,
                 **kwargs):
        """
        Constructor.

        Args:
            model: The model.
            dataset: The dataset.
        """
        super().__init__(model, dataset, **kwargs)

        self.D_iters = D_iters
        self.clamp = clamp
        self.gp_coeff = gp_coeff
        self.stats_interval = stats_interval

        # Initialize optimizers for generator and discriminator
        self.optims = {
            "applier": {
                "D": init_optim(self.model.applier.D.parameters(), **D_optim_config),
                "G": init_optim(self.model.applier.G.parameters(), **G_optim_config),
            },
            "remover": {
                "D": init_optim(self.model.remover.D.parameters(), **D_optim_config),
                "G": init_optim(self.model.remover.G.parameters(), **G_optim_config),
            },
        }


    def optims_zero_grad(self, D_or_G):
        """
        Zero gradients in all D optimizers or G optimizers.

        Args:
            D_or_G: Indicates whether the operation is for D optims or G optims.
                    Should be either "D" or "G".
        """
        [optim[D_or_G].zero_grad() for optim in self.optims if D_or_G in optim]


    def optims_step(self, D_or_G):
        """
        Make an optimization step in all D optimizers or G optimizers.

        Args:
            D_or_G: Indicates whether the operation is for D optims or G optims.
                    Should be either "D" or "G".
        """
        [optim[D_or_G].step() for optim in self.optims if D_or_G in optim]


    def train_step(self):
        """
        Makes ones training step.
        """

        print("Step: %d" % self.iters)

        ### Train D ###
        for _ in range(self.D_iters):
            # Sample from dataset
            sample = self.sample_dataset()
            # Unpack
            real_after = sample["after"].to(self.device)
            real_before = sample["before"].to(self.device)
            # Train
            D_loss = self.D_step(real_after, real_before)

        ### Train G ###
        # Sample from dataset
        sample = self.sample_dataset()
        # Unpack
        real_after = sample["after"].to(self.device)
        real_before = sample["before"].to(self.device)
        # Train
        G_loss = self.G_step(real_after, real_before)

        # Record data
        self.add_data(D_loss=D_loss, G_loss=G_loss)


    def D_step(self, real_after, real_before):

        # Sample from generators
        with torch.no_grad():
            fake_after = self.model.applier.G(real_before)
            fake_before = self.model.remover.G(real_after)

        # Zero gradients and loss
        self.optims_zero_grad("D")

        # Adversarial losses for after domain, before domain
        gan_configs = {"gan_type": self.model.gan_type, "gp_coeff": self.gp_coeff}
        D_loss = 0.1 * get_D_loss(self.model.applier.D, real_after, fake_after, **gan_configs)     \
               + 0.1 * get_D_loss(self.model.remover.D, real_before, fake_before, **gan_configs)

        # Calculate gradients
        D_loss.remover()

        # Make a step of minimizing D's loss
        self.optims_step("D")

        return D_loss


    def G_step(self, real_after, real_before):

        # Sample from generators
        fake_after = self.model.applier.G(real_before)
        fake_before = self.model.remover.G(real_after)

        # Zero gradients
        self.optims_zero_grad("G")

        # Adversarial loss for after domain, before domain
        gan_configs = {"gan_type": self.model.gan_type}
        G_loss = 0.1 * get_G_loss(self.model.applier.D, fake_after, **gan_configs)   \
               + 0.1 * get_G_loss(self.model.remover.D, fake_before, **gan_configs)

        # Identity loss
        G_loss += F.l1_loss(real_before, self.model.remover.G(fake_after))

        # Extra sparsity-inducing regularization
        G_loss += 0.1 * F.l1_loss(real_before, fake_after)

        # Calculate gradients
        G_loss.remover()

        # Make a step of minimizing G's loss
        self.optims_step("G")

        return G_loss




