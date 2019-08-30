
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from .losses import *
from .report_utils import *


class GAN_Trainer:
    """A trainer for a GAN."""

    def __init__(self, model, dataset,
        name="trainer",
        results_dir="results/",
        load_model_path=None,
        num_gpu=1,
        num_workers=2,
        batch_size=4,
        optimizer_name="sgd",
        lr=1e-4,
        momentum=0.9,
        betas=(0.5, 0.9),
        D_iters=5,
        clamp=0.01,
        gp_coeff=10.0,
        stats_report_interval=50,
        progress_check_interval=200,
        description="no description given"):
        """
        Initializes GAN_Trainer.

        Args:
            model: The makeup net.
            dataset: The makeup dataset.
            name: Name of this trainer.
            results_dir: Directory in which results will be saved for each run.
            load_model_path: Path to the model that will be loaded, if any.
            num_gpu: Number of GPUs to use for training.
            num_workers: Number of workers sampling from the dataset.
            batch_size: Size of the batch. Must be > num_gpu.
            optimizer_name: The name of the optimizer to use (e.g. "sgd").
            lr: The learning rate of the optimizer.
            momentum: The momentum of used in the optimizer, if applicable.
            betas: The betas used in the Adam optimizer.
            D_iters: Number of iterations to train discriminator every batch.
            clamp: Set to None if you don't want to clamp discriminator's weight after each update.
            gp_coeff: A coefficient for the gradient penalty (gp) of the discriminator.
            stats_report_interval: Report stats every `stats_report_interval` batch.
            progress_check_interval: Check progress every `progress_check_interval` batch.
            description: Description of the experiment the trainer is running.
        """

        # Initialize given parameters
        self.model = model
        self.dataset = dataset

        self.name = name
        self.results_dir = results_dir
        self.load_model_path = load_model_path
        
        self.num_gpu = num_gpu
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.betas = betas

        self.D_iters = D_iters
        self.clamp = clamp
        self.gp_coeff = gp_coeff

        self.stats_report_interval = stats_report_interval
        self.progress_check_interval = progress_check_interval

        self.description = description

        # Load model if necessary
        if load_model_path is not None:
            self.load_model(load_model_path)
        
        # Initialize device
        using_cuda = torch.cuda.is_available() and self.num_gpu > 0
        self.device = torch.device("cuda:0" if using_cuda else "cpu")

        # Move model to device and parallelize model if possible
        self.model = self.model.to(self.device)
        if self.device.type == "cuda" and self.num_gpu > 1:
            self.model = nn.DistributedDataParallel(self.model, list(range(self.num_gpu)))

        # Initialize optimizers for generator and discriminator
        self.D_optim = self.init_optim(self.model.D.parameters())
        self.G_optim = self.init_optim(self.model.G.parameters())

        # Initialize variables used for tracking loss and progress
        self.iters = 1  # current iteration
        self.losses = {"D": [-0.], "G": [-0.]}
        self.evals = {"D_on_real": [-0.], "D_on_fake1": [-0.], "D_on_fake2": [-0.]}
        self.generated_grids = []
        self.fixed_latent = torch.randn([64, self.model.num_latents], device=self.device)


    def init_optim(self, params):
        """
        Initializes the optimizer.

        Args:
            params: The parameters this optimizer will optimize.

        Returns:
            The optimizer (torch.optim). The default is SGD.
        """

        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(params, lr=self.lr, betas=self.betas)
        elif self.optimizer_name == "rmsprop":
            optim = torch.optim.RMSprop(params, lr=self.lr)
        elif self.optimizer_name == "sgd":
            optim = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        else:
            raise ValueError("Optimizer '{}' not recognized".format(self.optimizer_name))

        return optim


    def load_model(self, model_path):
        if not os.path.isfile(model_path):
            print(f"Couldn't load model: file '{model_path}' does not exist")
            print("Training model from scratch.")
        else:
            print("Loading model...")
            self.model.load_state_dict(torch.load(model_path))


    def save_model(self, model_path):
        print("Saving model")
        torch.save(self.model.state_dict(), model_path)


    #################### Training Methods ####################

    def run(self, num_epochs, save_results=False):
        """
        Runs the trainer. Trainer will train the model then save it.
        Note that running trainer more than once will accumulate the results.

        Args:
            num_epochs: Number of epochs to run.
            save_results: A flag indicating whether we should save the results this run.
        """

        # Initialize data loader
        data_loader = torch.utils.data.DataLoader(self.dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        try:
            # Try training the model
            self.train(data_loader, num_epochs)
        finally:
            # Stop trainer, report results
            self.stop(save_results)


    def train(self, data_loader, num_epochs):
        """
        Trains `self.model` on data loaded from data loader.

        Args:
            data_loader: An iterator from which the data is sampled.
            num_epochs: Number of epochs to run.
        """

        num_batches = len(data_loader)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(1, num_epochs + 1):
            for batch, sample in enumerate(data_loader, 1):

                # Train model on the samples
                self.train_on(sample)

                # Check progress of training so far
                self.checkpoint(epoch, num_epochs, batch, num_batches)

                self.iters += 1

        # Check progress of at the end of training
        self.checkpoint(num_batches, num_batches, num_epochs, num_epochs)
        print("Finished training.")


    def train_on(self, sample):
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

        real = sample["before"].to(self.device)
        real += 1e-3 * torch.randn_like(real)  # @XXX

        # Calculate latent vector size
        this_batch_size = real.size()[0]
        latent_size = torch.Size([this_batch_size, self.model.num_latents])

        # Sample fake images from a random latent vector
        latent = torch.randn(latent_size, device=self.device)
        fake = self.model.G(latent)

        # Train discriminator
        D_results = self.D_step(self.D_optim, self.model.D, real, fake.detach())
        D_loss, D_on_real, D_on_fake1 = D_results

        # Train generator if we trained discriminator D_iters time
        if self.iters % self.D_iters == 0:
            # Sample latent and fake again
            latent = torch.randn(latent_size, device=self.device)
            fake = self.model.G(latent)
            # Train generator
            G_results = self.G_step(self.G_optim, self.model.D, fake)
            G_loss, D_on_fake2 = G_results
        else:
            # @TODO: I'm sure there is a better way to handle this
            G_loss = self.losses["G"][-1]
            D_on_fake2 = self.evals["D_on_fake2"][-1]

        # Record losses and evaluations
        self.losses["D"].append(D_loss)
        self.losses["G"].append(G_loss)
        self.evals["D_on_real"].append(D_on_real)
        self.evals["D_on_fake1"].append(D_on_fake1)
        self.evals["D_on_fake2"].append(D_on_fake2)

        return D_on_real, D_on_fake1, D_on_fake2


    def D_step(self, D_optim, D, real, fake):
        """
        Trains the discriminator D and maximizes its score function.

        Args:
            D_optim: The optimizer for D.
            D: The discriminator.
            real: Real data point sampled from the dataset.
            fake: Fake data point sampled from the generator.

        Returns:
            A tuple containing the loss, the mean classification of D on
            the real images, as well as on the fake images, respectively.
        """

        gan_type = self.model.gan_type

        # Zero gradients
        D_optim.zero_grad()

        # Classify real and fake images
        D_on_real = D(real)
        D_on_fake = D(fake)

        if gan_type == "gan":
            D_loss = D_loss_GAN(D_on_real, D_on_fake)

        elif gan_type == "wgan":
            D_loss = D_loss_WGAN(D_on_real, D_on_fake)

        elif gan_type == "wgan-gp":
            D_grad_norm = D_grad_norm(D, real, fake)
            grad_penalty = D_grad_penalty(D_grad_norm, self.gp_coeff)
            D_loss = D_loss_WGAN(D_on_real, D_on_fake, grad_penalty=grad_penalty)

        else:
            raise ValueError(f"gan_type {gan_type} not supported")

        # Calculate gradients and minimize loss
        D_loss.backward()
        D_optim.step()

        # If WGAN, clamp D's weights to ensure k-Lipschitzness
        if gan_type == "wgan":
            [p.data.clamp_(*self.clamp) for p in D.parameters()]

        return (
            D_loss.item(),
            D_on_real.mean().item(),
            D_on_fake.mean().item(),
        )


    def G_step(self, G_optim, D, fake):
        """
        Trains the generator and minimizes its score function.

        Args:
            G_optim: The optimizer for G.
            D: The discriminator.
            fake: Fake data point generated from a latent variable.

        Returns:
            The mean loss of D on the fake images as well as
            the mean classification of the discriminator on the fake images.
        """

        gan_type = self.model.gan_type

        # Zero gradients
        G_optim.zero_grad()

        # Classify fake images
        D_on_fake = D(fake)

        if gan_type == "gan":
            G_loss = G_loss_GAN(D_on_fake)

        elif gan_type == "wgan" or gan_type == "wgan-gp":
            G_loss = G_loss_WGAN(D_on_fake)

        else:
            raise ValueError(f"gan_type {gan_type} not supported")

        # Calculate gradients and minimize loss
        G_loss.backward()
        G_optim.step()

        return G_loss.item(), D_on_fake.mean().item()


    #################### Reporting and Tracking Methods ####################

    def stop(self, save_results=False):
        """
        Stops the trainer and report the result of the experiment.

        Args:
            save_results: Results will be saved if this was set to True.
        """

        if not save_results:
            plot_lines(self.losses, title="Losses")
            plot_lines(self.evals, title="Evals")
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
        plot_lines(self.losses, losses_file, title="Losses of D and G")

        # Plot evals of D on real and fake data
        evals_file = os.path.join(experiment_dir, "evals.png")
        plot_lines(self.evals, evals_file, title="Evaluations of D on real and fake data")

        # Create an animation of the generator's progress
        animation_file = os.path.join(experiment_dir, "progress.mp4")
        create_progress_animation(self.generated_grids, animation_file)

        # Write description file of experiment
        description_txt = os.path.join(experiment_dir, "description.txt")
        with open(description_txt, "w") as f:
            f.write(self.description)


    def checkpoint(self, epoch, num_epochs, batch, num_batches):
        """
        The training checkpoint.

        Args:
            epoch: Current epoch.
            num_epochs: Number of epochs to run.
            batch: Current batch.
            num_batches: Number of batches to run.
        """

        # Report training stats
        if batch % self.stats_report_interval == 0:
            self.report_training_stats(batch, num_batches, epoch, num_epochs)

        # Check generator's progress by recording its output on a fixed input
        if self.iters % self.progress_check_interval == 0:
            grid = self.generate_grid(self.model.G, self.fixed_latent)
            self.generated_grids.append(grid)


    def generate_grid(self, generator, latent):
        """
        Check generator's output on latent vectors and return it.

        Args:
            generator: The generator.
            latent: Latent vector from which an image grid will be generated.

        Returns:
            A grid of images generated by `generator` from `latent`.
        """

        with torch.no_grad():
            fake = generator(latent).detach()

        image_grid = vutils.make_grid(fake.cpu(), padding=2, normalize=True)

        return image_grid


    def report_training_stats(self, epoch, num_epochs, batch, num_batches, precision=4):
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
            "D_loss": self.losses["D"][-1],
            "G_loss": self.losses["G"][-1],
            "D_of_x": self.evals["D_on_real"][-1],
            "D_of_G_z1": self.evals["D_on_fake1"][-1],
            "D_of_G_z2": self.evals["D_on_fake2"][-1],
            "p": precision,
        }

        print(report.format(**stats))


    def get_experiment_name(self, delimiter=", "):
        """
        Get the name of trainer's training train...

        Args:
            delimiter: The delimiter between experiment's parameters. Pretty useless.
        """

        experiment_details = {}

        # Train hyperparameters
        experiment_details["name"] = self.name
        experiment_details["iters"] = self.iters - 1
        experiment_details["batch_size"] = self.batch_size

        # Optimizer's hyperparameters
        experiment_details["optimizer_name"] = self.optimizer_name
        experiment_details["lr"] = self.lr
        if self.optimizer_name == "adam":
            experiment_details["betas"] = self.betas
        else:
            experiment_details["momentum"] = self.momentum

        # GAN's hyperparameters
        experiment_details["gan"] = self.model.gan_type
        experiment_details["D_iters"] = self.D_iters
        if self.model.gan_type == "wgan":
            experiment_details["clamp"] = self.clamp
        if self.model.gan_type == "wgan-gp":
            experiment_details["lambda"] = self.gp_coeff

        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        experiment = delimiter.join("{}={}".format(k,v) for k,v in experiment_details.items())

        return "[{}] {}".format(timestamp, experiment)

