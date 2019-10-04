
import os
import torch

from .base_trainer import BaseTrainer
from .utils.init_utils import init_optim
from .utils.gan_utils import *
from .utils.report_utils import *


class GAN_Trainer(BaseTrainer):
    """A trainer for a GAN."""

    def __init__(self, model, dataset,
        D_optim_config={},
        G_optim_config={},
        D_iters=5,
        clamp=0.01,
        gp_coeff=10.0,
        generate_grid_interval=200,
        **kwargs):
        """
        Initializes GAN_Trainer.

        Note:
            Optimizer's configurations/parameters must be passable to the
            optimizer (in torch.optim). It should also include a parameter
            `optim_choice` for the choice of the optimizer (e.g. "sgd" or "adam").

        Args:
            model: The model.
            dataset: The dataset.
            D_optim_config: Configurations for the discriminator's optimizer.
            G_optim_config: Configurations for the generator's optimizer.
            D_iters: Number of iterations to train discriminator every batch.
            clamp: Range on which the discriminator's weight will be clamped after each update.
            gp_coeff: A coefficient for the gradient penalty (gp) of the discriminator.
            generate_grid_interval: Check progress every `generate_grid_interval` batch.
        """
        super().__init__(model, dataset, **kwargs)

        self.D_iters = D_iters
        self.clamp = clamp
        self.gp_coeff = gp_coeff
        self.generate_grid_interval = generate_grid_interval

        # Initialize optimizers for generator and discriminator
        self.D_optim = init_optim(self.model.D.parameters(), **D_optim_config)
        self.G_optim = init_optim(self.model.G.parameters(), **G_optim_config)

        # Initialize list of image grids generated from a fixed latent variable
        grid_size = 8 * 8
        self._fixed_latent = torch.randn([grid_size, self.model.num_latents], device=self.device)
        self._generated_grids = []


    #################### Training Methods ####################

    def train_step(self, sample):
        """
        Trains on a sample of real and fake data.
        Throughout this file, we will denote a sample from the real data
        distribution, fake data distribution, and latent variables as:
            x ~ real,    x_g ~ fake,    z ~ latent

        Now recall that in order to train a GAN, we try to find a solution to
        a min-max game of the form `min_G max_D V(G,D)`, where G is the generator,
        D is the discriminator, and V(G,D) is the score function.
        For a regular GAN, V(G,D) = log(D(x)) + log(1 - D(x_g)),
        which is the Jensen-Shannon (JS) divergence between the probability
        distributions P(x) and P(x_g), where P(x_g) is parameterized by G.

        When it comes to Wasserstein GAN (WGAN), the objective is to minimize
        the Wasserstein (or Earth-Mover) distance instead of the JS-divergence.
        See Theorem 3 and Algorithm 1 in the original paper for more details.
        We can achieve that (thanks to the Kantorovich-Rubinstein duality)
        by first maximizing  `D(x) - D(x_g)` in the space of 1-Lipschitz
        discriminators D, where x ~ data and x_g ~ fake.
        Then, we have the gradient wrt G of the Wasserstein distance equal
        to the gradient of -D(G(z)).
        Since we assumed that D should be 1-Lipschitz, we can enforce
        k-Lipschitzness by clamping the weights of D to be in some fixed box,
        which would be approximate up to a scaling factor.

        Enforcing Lipschitzness is done more elegantly in WGAN-GP,
        which is just WGAN with gradient penalty (GP). The gradient penalty
        is used because of the statement that a differentiable function is
        1-Lipschitz iff it has gradient norm equal to 1 almost everywhere
        under P(x) and P(x_g). Hence, the objective will be similar to WGAN,
        which is `min_G max_D of D(x) - D(x_g)`, but now we add the gradient
        penalty in the D_step such that it will be minimized.

        Links to the papers:
        GAN:     https://arxiv.org/pdf/1406.2661.pdf
        WGAN:    https://arxiv.org/pdf/1701.07875.pdf
        WGAN-GP: https://arxiv.org/pdf/1704.00028.pdf


        Args:
            sample: Real data points sampled from the dataset.

        Returns:
            The result of the discriminator's classifications on real and fake data.
        """

        # Sample real data from the dataset
        real = sample["before"].to(self.device)
        batch_size = real.size()[0]

        # Train discriminator
        latent = self.sample_latent(batch_size)
        self.D_step(real, latent)

        # If WGAN, clamp D's weights to ensure k-Lipschitzness
        if self.model.gan_type == "wgan":
            [p.data.clamp_(*clamp) for p in D.parameters()]

        if self.iters % self.D_iters == 0:
            # Train generator if we trained discriminator D_iters times
            latent = self.sample_latent(batch_size)
            self.G_step(latent)
        else:
            # Fill the current iter's spot with the previous value
            self.add_data("G_loss", self.get_current_value("G_loss"))
            self.add_data("D_on_fake2", self.get_current_value("D_on_fake2"))


    def D_step(self, real, latent):

        D, G = self.model.D, self.model.G

        # Zero gradients
        self.D_optim.zero_grad()

        # Sample fake data from a latent (ignore gradients)
        with torch.no_grad():
            fake = G(latent).detach()  # @TODO: detach() redundant?

        # Classify real and fake data
        D_on_real = D(real)
        D_on_fake = D(fake)

        # Calculate loss and its gradients
        D_loss = get_D_loss(D, real, fake, gan_type=self.model.gan_type, gp_coeff=self.gp_coeff)
        D_loss.backward()

        # Calculate gradients and minimize loss
        self.D_optim.step()

        # Record results
        self.add_data("D_loss", D_loss.mean().item())
        self.add_data("D_on_real", D_on_real.mean().item())
        self.add_data("D_on_fake1", D_on_fake.mean().item())


    def G_step(self, latent):

        D, G = self.model.D, self.model.G
        
        # Zero gradients
        self.G_optim.zero_grad()

        # Sample fake data from latent
        fake = G(latent)

        # Classify fake data
        D_on_fake = D(fake)

        # Calculate loss and its gradients
        G_loss = get_G_loss(D, fake, gan_type=self.model.gan_type)
        G_loss.backward()

        # Optimize
        self.G_optim.step()

        # Record results
        self.add_data("G_loss", G_loss.mean().item())
        self.add_data("D_on_fake2", D_on_fake.mean().item())


    def sample_latent(self, batch_size):

        # Calculate latent size and sample from Gaussian
        latent_size = [batch_size, self.model.num_latents]
        latent = torch.randn(latent_size, device=self.device)

        return latent


    #################### Reporting and Tracking Methods ####################


    def stop(self, save_results=False):
        """
        Stops the trainer and report the result of the experiment.

        Args:
            save_results: Results will be saved if this was set to True.
        """

        losses = self.get_data_containing("loss")
        evals = self.get_data_containing("D_on")

        if not save_results:
            plot_lines(losses, title="Losses")
            plot_lines(evals, title="Evals")
            return

        # Create results directory if it hasn't been created yet
        if not os.path.isdir(self.results_dir): os.mkdir(self.results_dir)

        # Create experiment directory in the model's directory
        experiment_dir = os.path.join(self.results_dir, self.get_experiment_name())
        if not os.path.isdir(experiment_dir): os.mkdir(experiment_dir)

        # Save model
        model_path = os.path.join(experiment_dir, "model.pt")
        self.save_model(model_path)

        # Plot losses of D and G
        losses_file = os.path.join(experiment_dir, "losses.png")
        plot_lines(losses, losses_file, title="Losses of D and G")

        # Plot evals of D on real and fake data
        evals_file = os.path.join(experiment_dir, "evals.png")
        plot_lines(evals, evals_file, title="Evaluations of D on real and fake data")

        # Create an animation of the generator's progress
        animation_file = os.path.join(experiment_dir, "progress.mp4")
        create_progress_animation(self._generated_grids, animation_file)

        # Write details of experiment
        details_txt = os.path.join(experiment_dir, "repr.txt")
        with open(details_txt, "w") as f:
            f.write(self.__repr__())


    def post_train_step(self, epoch, num_epochs, batch, num_batches):
        """
        The post-training step.

        Args:
            epoch: Current epoch.
            num_epochs: Number of epochs to run.
            batch: Current batch.
            num_batches: Number of batches to run.
        """

        # Report training stats
        if batch % self.stats_interval == 0:
            self.report_training_stats(epoch, num_epochs, batch, num_batches)

        # Check generator's progress by recording its output on a fixed input
        if self.iters % self.generate_grid_interval == 0:
            grid = generate_grid(self.model.G, self._fixed_latent)
            self._generated_grids.append(grid)


    def _report_training_stats(self, epoch, num_epochs, batch, num_batches, precision=3):
        """
        Reports/prints the training stats to the console.

        Args:
            epoch: Current epoch.
            num_epochs: Max number of epochs.
            batch: Index of the current batch.
            num_batches: Max number of batches.
            precision: Precision of the float numbers reported.
        """

        report = \
            "[{epoch}/{num_epochs}][{batch}/{num_batches}]\t" \
            "Loss of D = {D_loss:.{p}f}\t" \
            "Loss of G = {G_loss:.{p}f}\t" \
            "D(x) = {D_of_x:.{p}f}\t" \
            "D(G(z)) = {D_of_G_z1:.{p}f} / {D_of_G_z2:.{p}f}"

        stats = {
            "epoch": epoch,
            "num_epochs": num_epochs,
            "batch": batch,
            "num_batches": num_batches,
            "D_loss": self.get_current_value("D_loss"),
            "G_loss": self.get_current_value("G_loss"),
            "D_of_x": self.get_current_value("D_on_real"),
            "D_of_G_z1": self.get_current_value("D_on_fake1"),
            "D_of_G_z2": self.get_current_value("D_on_fake2"),
            "p": precision,
        }

        print(report.format(**stats))

