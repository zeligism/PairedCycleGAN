
import os
import random
import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .utils.init_utils import init_optim
from .utils.gan_utils import *
from .utils.report_utils import *


class CycleGANTrainer(BaseTrainer):
    """The trainer for CycleGAN."""

    def __init__(self, model, dataset,
                 D_optim_config={},
                 G_optim_config={},
                 D_iters=5,
                 clamp=(-0.01, 0.01),
                 gp_coeff=10.,
                 generate_grid_interval=200,
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
        self.generate_grid_interval = generate_grid_interval

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

        # Generate makeup for a sample no-makeup faces and reference makeup faces
        num_test = 20
        self._generated_grids = []
        random_indices = random.sample(range(len(self.dataset)), num_test)
        self._fixed_before = torch.stack(
            [self.dataset[i]["before"] for i in random_indices], dim=0).to(self.device)


    def optims_zero_grad(self, D_or_G):
        """
        Zero gradients in all D optimizers or G optimizers.

        Args:
            D_or_G: Indicates whether the operation is for D optims or G optims.
                    Should be either "D" or "G".
        """
        [optim[D_or_G].zero_grad() for optim in self.optims.values() if D_or_G in optim]


    def optims_step(self, D_or_G):
        """
        Make an optimization step in all D optimizers or G optimizers.

        Args:
            D_or_G: Indicates whether the operation is for D optims or G optims.
                    Should be either "D" or "G".
        """
        [optim[D_or_G].step() for optim in self.optims.values() if D_or_G in optim]


    def train_step(self):
        """
        Makes ones training step.
        """

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
        self.writer.add_scalars("Loss", {"D_loss": D_loss, "G_loss": G_loss}, self.iters)


    def D_step(self, real_after, real_before):

        # Zero gradients and loss
        self.optims_zero_grad("D")

        # Sample from generators
        with torch.no_grad():
            fake_after = self.model.applier.G(real_before)
            fake_before = self.model.remover.G(real_after)

        # Adversarial losses for after domain, before domain
        gan_config = {"gan_type": self.model.gan_type, "gp_coeff": self.gp_coeff}
        D_loss = 0.1 * get_D_loss(self.model.applier.D, real_after, fake_after, **gan_config)     \
               + 0.1 * get_D_loss(self.model.remover.D, real_before, fake_before, **gan_config)

        # Calculate gradients
        D_loss.backward()

        # Make a step of minimizing D's loss
        self.optims_step("D")

        return D_loss


    def G_step(self, real_after, real_before):

        # Zero gradients
        self.optims_zero_grad("G")

        # Sample from generators
        fake_after = self.model.applier.G(real_before)
        fake_before = self.model.remover.G(real_after)

        # Adversarial loss for after domain, before domain
        gan_config = {"gan_type": self.model.gan_type}
        G_loss = 0.1 * get_G_loss(self.model.applier.D, fake_after, **gan_config)   \
               + 0.1 * get_G_loss(self.model.remover.D, fake_before, **gan_config)

        # Identity loss
        G_loss += F.l1_loss(real_before, self.model.remover.G(fake_after))

        # Extra sparsity-inducing regularization
        G_loss += 0.1 * F.l1_loss(real_before, fake_after)

        # Calculate gradients
        G_loss.backward()

        # Make a step of minimizing G's loss
        self.optims_step("G")

        return G_loss


    #################### Reporting and Tracking Methods ####################

    def post_train_step(self):
        """
        The post-training step.
        """
        super().post_train_step()

        should_generate_grid = self.iters % self.generate_grid_interval == 0

        # Check generator's progress by recording its output on a fixed input
        if should_generate_grid:
            grid = generate_applier_grid(self.model.applier.G, self._fixed_before)
            self._generated_grids.append(grid)
            self.writer.add_image("grid", grid, self.iters)


    def stop(self):
        """
        Stops the trainer and report the result of the experiment.
        """

        losses = self.get_data_containing("loss")

        if not self.save_results:
            plot_lines(losses, title="Losses")
            return

        # Create experiment directory in the model's directory
        experiment_dir = os.path.join(self.results_dir, self.get_experiment_name())

        # Save model
        model_path = os.path.join(experiment_dir, "model.pt")
        self.save_model(model_path)

        # Plot losses of D and G
        losses_file = os.path.join(experiment_dir, "losses.png")
        plot_lines(losses, filename=losses_file, title="Losses of D and G")

        # Create an animation of the generator's progress
        animation_file = os.path.join(experiment_dir, "progress.mp4")
        create_progress_animation(self._generated_grids, animation_file)

        # Write details of experiment
        details_txt = os.path.join(experiment_dir, "repr.txt")
        with open(details_txt, "w") as f:
            f.write(self.__repr__())




