
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

    def train_step(self):
        """
        Makes one training step.
        Throughout this doc, we will denote a sample from the real data
        distribution, fake data distribution, and latent variables respectively
        as follows:
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
        """

        for _ in range(self.D_iters):
            # Sample real data from the dataset
            sample = self.sample_dataset()
            real = sample["before"].to(self.device)

            # Sample latent and train discriminator
            latent = self.sample_latent()
            D_results = self.D_step(real, latent)

        # Sample latent and train generator
        latent = self.sample_latent()
        G_results = self.G_step(latent)
        
        # Record data
        self.add_data(**D_results, **G_results)


    def D_step(self, real, latent):
        """
        Makes a training step for the discriminator of the model.

        Args:
            real: Sample from the dataset.
            latent: Sample from the latent space.

        Returns:
            D loss and evaluation of D on real and on fake.
        """

        D, G = self.model.D, self.model.G

        # Zero gradients
        self.D_optim.zero_grad()

        # Sample fake data from a latent (ignore gradients)
        with torch.no_grad():
            fake = G(latent)

        # Classify real and fake data
        D_on_real = D(real)
        D_on_fake = D(fake)

        # Calculate loss and its gradients
        D_loss = get_D_loss(D, real, fake, gan_type=self.model.gan_type, gp_coeff=self.gp_coeff)
        D_loss.backward()

        # Calculate gradients and minimize loss
        self.D_optim.step()

        # If WGAN, clamp D's weights to ensure k-Lipschitzness
        if self.model.gan_type == "wgan":
            [p.data.clamp_(*clamp) for p in D.parameters()]

        return {
            "D_loss": D_loss.mean().item(),
            "D_on_real": D_on_real.mean().item(),
            "D_on_fake1": D_on_fake.mean().item()
        }


    def G_step(self, latent):
        """
        Makes a training step for the generator of the model.

        Args:
            latent: Sample from the latent space.
        
        Returns:
            G loss and evaluation of D on fake.
        """

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
        return {
            "G_loss": G_loss.mean().item(),
            "D_on_fake2": D_on_fake.mean().item(),
        }


    def sample_latent(self):
        """
        Samples from the latent space (i.e. input space of the generator).

        Returns:
            Sample from the latent space.
        """

        # Calculate latent size and sample from normal distribution
        latent_size = [self.batch_size, self.model.num_latents]
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

        # Create experiment directory in the model's directory
        experiment_dir = os.path.join(self.results_dir, self.get_experiment_name())
        if not os.path.isdir(experiment_dir): os.mkdir(experiment_dir)

        # Save model
        model_path = os.path.join(experiment_dir, "model.pt")
        self.save_model(model_path)

        # Plot losses of D and G
        losses_file = os.path.join(experiment_dir, "losses.png")
        plot_lines(losses, filename=losses_file, title="Losses of D and G")

        # Plot evals of D on real and fake data
        evals_file = os.path.join(experiment_dir, "evals.png")
        plot_lines(evals, filename=evals_file, title="Evaluations of D on real and fake data")

        # Create an animation of the generator's progress
        animation_file = os.path.join(experiment_dir, "progress.mp4")
        create_progress_animation(self._generated_grids, animation_file)

        # Write details of experiment
        details_txt = os.path.join(experiment_dir, "repr.txt")
        with open(details_txt, "w") as f:
            f.write(self.__repr__())


    def post_train_step(self):
        """
        The post-training step.
        """

        should_report_stats = self.iters % self.stats_interval == 0
        should_generate_grid = self.iters % self.generate_grid_interval == 0
        finished_epoch = self.batch == self.num_batches

        # Report training stats
        if should_report_stats or finished_epoch:
            self.report_training_stats()

        # Check generator's progress by recording its output on a fixed input
        if should_generate_grid:
            grid = generate_grid(self.model.G, self._fixed_latent)
            self._generated_grids.append(grid)


    def report_training_stats(self, precision=3):
        """
        Reports/prints the training stats to the console.

        Args:
            precision: Precision of the float numbers reported.
        """

        report = \
            "[{epoch}/{num_epochs}][{batch}/{num_batches}]\t" \
            "Loss of D = {D_loss:.{p}f}\t" \
            "Loss of G = {G_loss:.{p}f}\t" \
            "D(x) = {D_on_real:.{p}f}\t" \
            "D(G(z)) = {D_on_fake1:.{p}f} / {D_on_fake2:.{p}f}"

        stats = {
            "epoch": self.epoch,
            "num_epochs": self.num_epochs,
            "batch": self.batch,
            "num_batches": self.num_batches,
            "D_loss": self.get_current_value("D_loss"),
            "G_loss": self.get_current_value("G_loss"),
            "D_on_real": self.get_current_value("D_on_real"),
            "D_on_fake1": self.get_current_value("D_on_fake1"),
            "D_on_fake2": self.get_current_value("D_on_fake2"),
            "p": precision,
        }

        print(report.format(**stats))

