
import torch
import torch.nn.functional as F

from .gan_trainer import GAN_Trainer
from .base_trainer import BaseTrainer
from .utils.init_utils import init_optim
from .utils.gan_utils import *
from .utils.face_morph.face_morph import face_morph


class _MakeupNetTrainer(GAN_Trainer):
    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)


class MakeupNetTrainer(BaseTrainer):
    """The trainer for MakeupNet."""

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
            model: The makeup net.
            dataset: The makeup dataset.
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
            "style": {
                "D": init_optim(self.model.style_D.parameters(), **D_optim_config),
                # "G": init_optim([]),
            }
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

        Args:
            sample: A sample from the dataset.
        """

        print("Step: %d" % self.iters)

        ### Train D ###
        for _ in range(self.D_iters):
            # Sample from dataset
            sample = self.sample_dataset()
            # Unpack
            real_makeup = sample["after"].to(self.device)
            real_nomakeup = sample["before"].to(self.device)
            makeup_lm = sample["landmarks"]["after"]
            nomakeup_lm = sample["landmarks"]["before"]
            # Train
            D_loss = self.D_step(real_makeup, real_nomakeup, makeup_lm, nomakeup_lm)

        ### Train G ###
        # Sample from dataset
        sample = self.sample_dataset()
        # Unpack
        real_makeup = sample["after"].to(self.device)
        real_nomakeup = sample["before"].to(self.device)
        # Train
        G_loss = self.G_step(real_makeup, real_nomakeup)

        # Record data
        self.add_data(D_loss=D_loss, G_loss=G_loss)


    def D_step(self, real_makeup, real_nomakeup, makeup_lm, nomakeup_lm):

        # Sample from generators
        with torch.no_grad():
            fake_makeup = self.model.applier.G(real_nomakeup, real_makeup)
            fake_nomakeup = self.model.remover.G(real_makeup)

        real_styles = self.sample_real_styles(real_makeup, real_nomakeup, makeup_lm, nomakeup_lm)
        fake_styles = self.sample_fake_styles(real_makeup, fake_makeup)

        # Zero gradients and loss
        self.optims_zero_grad("D")

        # Adversarial losses for makeup domain, no-makeup domain, and styles domain
        gan_configs = {"gan_type": self.model.gan_type, "gp_coeff": self.gp_coeff}
        D_loss = 0.1 * get_D_loss(self.model.applier.D, real_makeup, fake_makeup, **gan_configs)     \
               + 0.1 * get_D_loss(self.model.remover.D, real_nomakeup, fake_nomakeup, **gan_configs) \
               + 0.1 * get_D_loss(self.model.style_D, real_styles, fake_styles, **gan_configs)

        # Calculate gradients
        D_loss.backward()

        # Make a step of minimizing D's loss
        self.optims_step("D")

        return D_loss


    def G_step(self, real_makeup, real_nomakeup):

        # Sample from generators
        fake_makeup = self.model.applier.G(real_nomakeup, real_makeup)
        fake_nomakeup = self.model.remover.G(real_makeup)

        fake_styles = self.sample_fake_styles(real_makeup, fake_makeup)

        # Zero gradients
        self.optims_zero_grad("G")

        # Adversarial loss for makeup domain, no-makeup domain, and style domain
        gan_configs = {"gan_type": self.model.gan_type}
        G_loss = 0.1 * get_G_loss(self.model.applier.D, fake_makeup, **gan_configs)   \
               + 0.1 * get_G_loss(self.model.remover.D, fake_nomakeup, **gan_configs) \
               + 0.1 * get_G_loss(self.model.style_D, fake_styles, **gan_configs)

        # Identity loss
        G_loss += F.l1_loss(real_nomakeup, self.model.remover.G(fake_makeup))

        # Style loss (i.e. style is preserved in fake_makeup and well-removed in fake_nomakeup)
        G_loss += F.l1_loss(real_makeup, self.model.applier.G(fake_nomakeup, fake_makeup))

        # Extra sparsity-inducing regularization for makeup mask
        G_loss += 0.1 * F.l1_loss(real_nomakeup, fake_makeup)

        # Calculate gradients
        G_loss.backward()

        # Make a step of minimizing G's loss
        self.optims_step("G")

        return G_loss


    def sample_real_styles(self, real_makeup, real_nomakeup, makeup_lm, nomakeup_lm):
        # Morph makeup face to nomakeup face's facial structure for style loss calculation
        mask, morphed_real_makeup = self.morph_makeup(real_makeup, real_nomakeup,
                                                      makeup_lm, nomakeup_lm)

        # Prepare real same style pair vs. fake same style pair
        real_styles = torch.cat([mask * real_makeup , mask * morphed_real_makeup], dim=1)

        return real_styles


    def sample_fake_styles(self, real_makeup, fake_makeup):
        fake_styles = torch.cat([real_makeup , fake_makeup], dim=1)

        return fake_styles


    def morph_makeup(self, real_makeup, real_nomakeup, makeup_lm, nomakeup_lm):

        tensor2D_to_points = lambda t: [(p[0].item(), p[1].item()) for p in t]
        torch_to_numpy = lambda t: t.permute(1, 2, 0).numpy()
        numpy_to_torch = lambda t: torch.from_numpy(t).permute(2, 0, 1)

        batch_size = real_makeup.size()[0]
        mask = torch.ones([batch_size, 1, 1, 1]).to(real_makeup)
        morphed_batch = []

        for i in range(batch_size):
            # Zero mask for no landmarks
            if makeup_lm[i].sum() == 0 or nomakeup_lm[i].sum() == 0:
                morphed_batch.append(torch.zeros_like(real_nomakeup[i]))
                mask[i] = 0
            else:
                morphed = face_morph(torch_to_numpy(real_makeup[i]),
                                     torch_to_numpy(real_nomakeup[i]),
                                     tensor2D_to_points(makeup_lm[i]),
                                     tensor2D_to_points(nomakeup_lm[i]))
                morphed_batch.append(numpy_to_torch(morphed))

        return mask, torch.stack(morphed_batch).to(real_makeup)




