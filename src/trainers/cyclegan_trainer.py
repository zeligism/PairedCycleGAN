
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
                 before_noise_std=0.01,
                 after_noise_std=0.01,
                 generate_grid_interval=200,
                 constants={},
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
        self.before_noise_std = before_noise_std
        self.after_noise_std  = after_noise_std
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

        # TODO: Specify loss type instead (minimax or wasserstein)
        self.D_loss_fn = get_D_loss(self.model.gan_type)
        self.G_loss_fn = get_G_loss(self.model.gan_type)

        # Initialize all constants required for training
        self.constants = self._get_constants(**constants)

        # Generate makeup for a sample no-makeup faces and reference makeup faces
        num_test = 20
        self._applier_generated_grids = []
        random_indices = random.sample(range(len(self.dataset)), num_test)
        self._fixed_before = torch.stack(
            [self.dataset[i]["before"] for i in random_indices], dim=0).to(self.device)
        
        self._remover_generated_grids = []
        random_indices = random.sample(range(len(self.dataset)), num_test)
        self._fixed_after = torch.stack(
            [self.dataset[i]["after"] for i in random_indices], dim=0).to(self.device)


    def _get_constants(self,
                       applier_adversarial=0.1,
                       remover_adversarial=0.1,
                       applier_D_grad_penalty=0.,
                       remover_D_grad_penalty=0.,
                       after_identity_robustness=0.0,
                       before_identity_robustness=0.1,
                       applier_mask_sparsity=0.1,
                       remover_mask_sparsity=0.0,
                       **kwargs):
        return {
            "applier_adversarial": applier_adversarial,
            "remover_adversarial": remover_adversarial,
            "applier_D_grad_penalty": applier_D_grad_penalty,
            "remover_D_grad_penalty": remover_D_grad_penalty,
            "after_identity_robustness": after_identity_robustness,
            "before_identity_robustness": before_identity_robustness,
            "applier_mask_sparsity": applier_mask_sparsity,
            "remover_mask_sparsity": remover_mask_sparsity,
        }

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
            D_results = self.D_step(real_after, real_before)

        ### Train G ###
        # Sample from dataset
        sample = self.sample_dataset()
        # Unpack
        real_after = sample["after"].to(self.device)
        real_before = sample["before"].to(self.device)
        # Train
        G_results = self.G_step(real_after, real_before)

        # Record data
        self.add_data(**D_results, **G_results)
        losses = {"D_loss": D_results["D_loss"], "G_loss": G_results["G_loss"]}
        self.writer.add_scalars("Loss", losses, self.iters)


    def D_step(self, real_after, real_before):

        # Zero gradients and loss
        self.optims_zero_grad("D")

        # Sample noise
        noise_after = torch.randn_like(real_after) * self.after_noise_std
        noise_before = torch.randn_like(real_before) * self.before_noise_std

        # Add noise to real
        real_after += noise_after
        real_before += noise_before

        # Sample from generators
        with torch.no_grad():
            fake_after = self.model.applier.G(real_before)
            fake_before = self.model.remover.G(real_after)
        
        # Add noise to fake
        fake_after += noise_after
        fake_before += noise_before

        # Classify real and fake images
        remover_D_on_real = self.model.remover.D(real_before)
        remover_D_on_fake = self.model.remover.D(fake_before)
        applier_D_on_real = self.model.applier.D(real_after)
        applier_D_on_fake = self.model.applier.D(fake_after)

        # Adversarial losses for after domain, before domain
        remover_adv_loss = self.D_loss_fn(remover_D_on_real, remover_D_on_fake)
        applier_adv_loss = self.D_loss_fn(applier_D_on_real, applier_D_on_fake)

        # Gradient penalty XXX ?
        applier_D_grad_penalty = torch.tensor(0.0)
        if self.constants["applier_D_grad_penalty"] > 0:
            interpolated_after = random_interpolate(real_after, fake_after)
            applier_D_grad_penalty = simple_gradient_penalty(self.model.applier.D, interpolated_after, center=1.0)

        remover_D_grad_penalty = torch.tensor(0.0)
        if self.constants["remover_D_grad_penalty"] > 0:
            interpolated_before = random_interpolate(real_before, fake_before)
            remover_D_grad_penalty = simple_gradient_penalty(self.model.remover.D, interpolated_before, center=1.0)
        
        # Calculate gradients and minimize loss
        D_loss = self.constants["applier_adversarial"] * applier_adv_loss \
               + self.constants["remover_adversarial"] * remover_adv_loss \
               + self.constants["applier_D_grad_penalty"] * applier_D_grad_penalty \
               + self.constants["remover_D_grad_penalty"] * remover_D_grad_penalty
        D_loss.backward()
        self.optims_step("D")

        return {
            "applier_D_on_real": applier_D_on_real.mean().item(),
            "remover_D_on_real": remover_D_on_real.mean().item(),
            "applier_D_on_fake": applier_D_on_fake.mean().item(),
            "remover_D_on_fake": remover_D_on_fake.mean().item(),
            "applier_D_grad_penalty": applier_D_grad_penalty.item(),
            "remover_D_grad_penalty": remover_D_grad_penalty.item(),
            "D_loss": D_loss.item(),
        }


    def G_step(self, real_after, real_before):

        # Zero gradients
        self.optims_zero_grad("G")

        # Sample noise
        noise_after = torch.randn_like(real_after) * self.after_noise_std
        noise_before = torch.randn_like(real_before) * self.before_noise_std

        # Add noise to real
        real_after += noise_after
        real_before += noise_before

        # Sample from generators
        fake_after = self.model.applier.G(real_before)
        fake_before = self.model.remover.G(real_after)

        # Add noise to fake
        fake_after += noise_after
        fake_before += noise_before

        # Classify fake images
        applier_D_on_fake = self.model.applier.D(fake_after)
        remover_D_on_fake = self.model.remover.D(fake_before)

        # Adversarial loss for after domain, before domain
        remover_adv_loss = self.G_loss_fn(remover_D_on_fake)
        applier_adv_loss = self.G_loss_fn(applier_D_on_fake)

        # Identity loss for applier.D's domain (after)
        after_identity_loss = torch.tensor(0.0)
        if self.constants["after_identity_robustness"] > 0:
            after_identity_loss = F.l1_loss(real_after, self.model.applier.G(fake_before))
        # Identity loss for remover.D's domain (before)
        before_identity_loss = torch.tensor(0.0)
        if self.constants["before_identity_robustness"] > 0:
            before_identity_loss = F.l1_loss(real_before, self.model.remover.G(fake_after))

        # Sparsity regularization for applier
        applier_sparsity_loss = torch.tensor(0.0)
        if self.constants["applier_mask_sparsity"] > 0:
            applier_sparsity_loss = F.l1_loss(real_before, fake_after)
        # Sparsity regularization for remover
        remover_sparsity_loss = torch.tensor(0.0)
        if self.constants["remover_mask_sparsity"] > 0:
            remover_sparsity_loss = F.l1_loss(real_after, fake_before)

        # Calculate gradients and minimize loss
        G_loss = self.constants["applier_adversarial"] * applier_adv_loss \
               + self.constants["remover_adversarial"] * remover_adv_loss \
               + self.constants["before_identity_robustness"] * before_identity_loss \
               + self.constants["after_identity_robustness"] * after_identity_loss \
               + self.constants["applier_mask_sparsity"] * applier_sparsity_loss \
               + self.constants["remover_mask_sparsity"] * remover_sparsity_loss
        G_loss.backward()
        self.optims_step("G")

        return {
            "applier_D_on_fake2": applier_D_on_fake.mean().item(),
            "remover_D_on_fake2": remover_D_on_fake.mean().item(),
            "before_identity_loss": before_identity_loss.item(),
            "after_identity_loss": after_identity_loss.item(),
            "applier_sparsity_loss": applier_sparsity_loss.item(),
            "remover_sparsity_loss": remover_sparsity_loss.item(),
            "G_loss": G_loss.item(),
        }


    #################### Reporting and Tracking Methods ####################

    def post_train_step(self):
        """
        The post-training step.
        """
        super().post_train_step()

        should_generate_grid = self.iters % self.generate_grid_interval == 0

        # Check generator's progress by recording its output on a fixed input
        if should_generate_grid:
            applier_grid = generate_G_grid(self.model.applier.G, self._fixed_before)
            remover_grid = generate_G_grid(self.model.remover.G, self._fixed_after)
            self._applier_generated_grids.append(applier_grid)
            self._remover_generated_grids.append(remover_grid)
            self.writer.add_image("Grids/applier_grid", applier_grid, self.iters)
            self.writer.add_image("Grids/remover_grid", remover_grid, self.iters)


    def stop(self, lines_to_plot={}):
        """
        Stops the trainer and report the result of the experiment.
        """

        losses = {**self.get_data_containing("D_loss"), **self.get_data_containing("G_loss")}

        # XXX
        lines_to_plot = {
            "Discriminator Evaluations": "D_on",
            "Gradient Penalty": "grad_penalty",
            "Sparsity Loss": "sparsity",
            "Identity Loss": "identity",
        }

        if not self.save_results:
            plot_lines(losses, title="Losses")
            for title, keyword in lines_to_plot.items():
                plot_lines(self.get_data_containing(keyword), title=title)
            return

        # Create experiment directory in the model's directory
        experiment_dir = os.path.join(self.results_dir, self.get_experiment_name())

        # Save model
        model_path = os.path.join(experiment_dir, "model.pt")
        self.save_model(model_path)

        # Plot losses of D and G
        losses_file = os.path.join(experiment_dir, "Losses.png")
        plot_lines(losses, filename=losses_file, title="Losses")
        for title, keyword in lines_to_plot.items():
            line_file = os.path.join(experiment_dir, f"{title}.png")
            plot_lines(self.get_data_containing(keyword), filename=line_file, title=title)

        # Create an animation of the generator's progress
        remover_animation_file = os.path.join(experiment_dir, "remover_progress.mp4")
        applier_animation_file = os.path.join(experiment_dir, "applier_progress.mp4")
        create_progress_animation(self._remover_generated_grids, remover_animation_file)
        create_progress_animation(self._applier_generated_grids, applier_animation_file)

        # Write details of experiment
        details_txt = os.path.join(experiment_dir, "repr.txt")
        with open(details_txt, "w") as f:
            f.write(self.__repr__())

