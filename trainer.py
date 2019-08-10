
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from report_utils import plot_lines, create_progress_animation


class MakeupNetTrainer:
    """The trainer for MakeupNet."""

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
        gan_type="gan",
        D_iters=5,
        clamp=0.01,
        gp_coeff=10.0,
        stats_report_interval=50,
        progress_check_interval=200):
        """
        Initializes MakeupNetTrainer.

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
            gan_type: Type of GAN to train. Choices = {gan, wgan, wgan-gp}
            D_iters: Number of iterations to train discriminator every batch.
            clamp: Set to None if you don't want to clamp discriminator's weight after each update.
            gp_coeff: A coefficient for the gradient penalty (gp) of the discriminator.
            stats_report_interval: Report stats every `stats_report_interval` batch.
            progress_check_interval: Check progress every `progress_check_interval` batch.
        """

        # Initialize given parameters
        self.model = model
        self.dataset = dataset

        self.name = name
        self.results_dir = results_dir
        #self.load_model_path = load_model_path
        
        self.num_gpu = num_gpu
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.betas = betas

        self.gan_type = gan_type
        self.D_iters = D_iters
        self.clamp = clamp
        self.gp_coeff = gp_coeff

        self.stats_report_interval = stats_report_interval
        self.progress_check_interval = progress_check_interval

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
        self.grad_norms = []
        self.progress_frames = []
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
            for batch_index, sample in enumerate(data_loader, 1):

                # Sample real images
                real = sample["before"].to(self.device)

                # Train model on the samples, and get the the discriminator's results
                self.train_on(real)

                # Report training stats
                if batch_index % self.stats_report_interval == 0:
                    self.report_training_stats(batch_index, num_batches, epoch, num_epochs)

                # Check generator's progress by recording its output on a fixed input
                if self.iters % self.progress_check_interval == 0:
                    self.check_progress_of_generator()

                self.iters += 1

        # Show stats and check progress at the end
        self.report_training_stats(num_batches, num_batches, num_epochs, num_epochs)
        self.check_progress_of_generator()
        print("Finished training.")


    def train_on(self, real):
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
            real: Real data points sampled from the dataset.

        Returns:
            The result of the discriminator's classifications on real and fake data.
        """

        real += 1e-3 * torch.randn_like(real)  # @XXX

        # Calculate latent vector size
        this_batch_size = real.size()[0]
        latent_size = torch.Size([this_batch_size, self.model.num_latents])

        # Sample fake images from a random latent vector
        latent = torch.randn(latent_size, device=self.device)
        fake = self.model.G(latent)
        # Train discriminator
        D_loss, D_on_real, D_on_fake1 = self.D_step(real, fake)

        # Train generator if we trained discriminator D_iters time
        if self.iters % self.D_iters == 0:
            # Sample latent and fake again
            latent = torch.randn(latent_size, device=self.device)
            fake = self.model.G(latent)
            G_loss, D_on_fake2 = self.G_step(fake)
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


    def D_step(self, real, fake):
        """
        Trains the discriminator. Maximize the score function.

        Args:
            real: The real image sampled from the dataset.
            fake: The fake image generated by the generator.

        Returns:
            A tuple containing the loss, the mean classification of D on
            the real images, as well as on the fake images, respectively.
        """

        batch_size = real.size()[0]

        # Zero gradients
        self.D_optim.zero_grad()

        # Classify real and fake images
        D_on_real = self.model.D(real)
        D_on_fake = self.model.D(fake.detach())  # don't pass grads to G

        if self.gan_type == "gan":
            # Create (noisy) real and fake labels
            real_label = 0.8 + 0.2 * torch.rand([batch_size], device=self.device)
            fake_label = 0.05 * torch.rand([batch_size], device=self.device)
            # Calculate binary cross entropy loss
            D_loss_on_real = F.binary_cross_entropy_with_logits(D_on_real, real_label)
            D_loss_on_fake = F.binary_cross_entropy_with_logits(D_on_fake, fake_label)
            # Loss is: - log(D(x)) - log(1 - D(x_g)),
            # which is equiv. to maximizing: log(D(x)) + log(1 - D(x_g))
            D_loss = torch.mean(D_loss_on_real + D_loss_on_fake)

        elif self.gan_type == "wgan":
            # Maximize: D(x) - D(x_g), i.e. minimize -(D(x) - D(x_g))
            D_loss = -1 * torch.mean(D_on_real - D_on_fake)

        elif self.gan_type == "wgan-gp":
            # Calculate gradient penalty
            eps = torch.rand(real.size(), device=self.device)
            interpolated = (1 - eps) * fake.detach() + eps * real
            interpolated.requires_grad_()
            D_on_inter = self.model.D(interpolated)
            # Calculate gradient of D(x_i) wrt x_i for each batch
            D_grad = torch.autograd.grad(D_on_inter, interpolated,
                                         torch.ones_like(D_on_inter), retain_graph=True)
            # D_grad will be a 1-tuple, as in: (grad,)
            D_grad_norm = D_grad[0].view([batch_size, -1]).norm(dim=1)
            grad_penalty = self.gp_coeff * (D_grad_norm - 1).pow(2)
            # Maximize: D(x) - D(x_g) - gp_coeff * (|| grad of D(x_i) wrt x_i || - 1)^2,
            # where x_i <- eps * x + (1 - eps) * x_g, and eps ~ rand(0,1)
            D_loss = -1 * torch.mean(D_on_real - D_on_fake - grad_penalty)
            self.grad_norms.append(D_grad_norm.mean().item())

        else:
            raise ValueError(f"gan_type {self.gan_type} not supported")

        # Calculate gradients and minimize loss
        D_loss.backward()
        self.D_optim.step()

        # If WGAN, clamp D's weights to ensure k-Lipschitzness
        if self.gan_type == "wgan":
            [p.data.clamp_(*self.clamp) for p in self.model.D.parameters()]

        return (
            D_loss.item(),
            D_on_real.mean().item(),
            D_on_fake.mean().item(),
        )


    def G_step(self, fake):
        """
        Trains the generator. Minimize the score function.

        Args:
            fake: The fake image generated by the generator.

        Returns:
            The mean loss of D on the fake images as well as
            the mean classification of the discriminator on the fake images.
        """

        # Zero gradients
        self.G_optim.zero_grad()

        # Classify fake images
        D_on_fake = self.model.D(fake)

        if self.gan_type == "gan":
            # Calculate binary cross entropy loss with a fake binary label
            batch_size = fake.size()[0]
            fake_label = torch.zeros([batch_size], device=self.device)
            # Loss is: -log(D(G(z))), which is equiv. to minimizing log(1-D(G(z)))
            # We use this loss vs. the original one for stability only.
            G_loss = F.binary_cross_entropy_with_logits(D_on_fake, 1 - fake_label)

        elif self.gan_type == "wgan" or self.gan_type == "wgan-gp":
            # Minimize: -D(G(z))
            G_loss = (-D_on_fake).mean()

        else:
            raise ValueError(f"gan_type {self.gan_type} not supported")

        # Calculate gradients and minimize loss
        G_loss.backward()
        self.G_optim.step()

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
        create_progress_animation(self.progress_frames, animation_file)


    def check_progress_of_generator(self):
        """
        Check generator's output on a fixed latent vector and record it.
        """

        with torch.no_grad():
            fixed_fake = self.model.G(self.fixed_latent).detach()

        progress_grid = vutils.make_grid(fixed_fake.cpu(), padding=2, normalize=True)

        self.progress_frames.append(progress_grid)


    def report_training_stats(self, batch_index, num_batches, epoch, num_epochs, precision=4):
        """
        Reports/prints the training stats to the console.

        Args:
            batch_index: Index of the current batch.
            num_batches: Max number of batches.
            epoch: Current epoch.
            num_epochs: Max number of epochs.
            precision: Precision of the float numbers reported.
        """

        report = \
            "[{epoch}/{num_epochs}][{batch_index}/{num_batches}]\t" \
            "Loss of D = {D_loss:.{p}f}\t" \
            "Loss of G = {G_loss:.{p}f}\t" \
            "D(x) = {D_of_x:.{p}f}\t" \
            "D(G(z)) = {D_of_G_z1:.{p}f} / {D_of_G_z2:.{p}f}"

        stats = {
            "epoch": epoch,
            "num_epochs": num_epochs,
            "batch_index": batch_index,
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
        experiment_details["gan"] = self.gan_type
        experiment_details["D_iters"] = self.D_iters
        if self.gan_type == "wgan":
            experiment_details["clamp"] = self.clamp
        if self.gan_type == "wgan-gp":
            experiment_details["lambda"] = self.gp_coeff

        timestamp = "[{}]".format(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        experiment = delimiter.join("{}={}".format(k,v) for k,v in experiment_details.items())

        return timestamp + " " + experiment

