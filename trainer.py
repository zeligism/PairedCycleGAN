
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

EPSILON = 1e-4
REAL = 1.0 - EPSILON
FAKE = 0.0 + EPSILON


class MakeupNetTrainer:
	"""The trainer for MakeupNet."""

	def __init__(self, model, dataset, name="trainer",
		load_model=False, model_path="model.pt",
		num_gpu=0, num_epochs=5, batch_size=4,
		optimizer_name="sgd", lr=1e-4, momentum=0.9,
		stats_report_interval=50, progress_check_interval=50):
		"""
		Initializes MakeupNetTrainer.

		Args:
			model: The makeup net.
			dataset: The makeup dataset.
			name: Name of this trainer.
			load_model: A flag indicating whether we should load the model or not.
			model_path: The path to the file of the model.
			num_gpu: Number of GPUs to use for training.
			num_epochs: Number of epochs to train.
			batch_size: Size of the batch. Must be > num_gpu.
			optimizer_name: The name of the optimizer to use (e.g. "sgd").
			lr: The learning rate of the optimizer.
			momentum: The momentum of used in the optimizer, if applicable.
			stats_report_interval: Report stats every `stats_report_interval` batch.
			progress_check_interval: Check progress every `progress_check_interval` batch.
		"""

		# Initialize given parameters
		self.model = model
		self.dataset = dataset
		self.name = name
		self.load_model = load_model
		self.model_path = model_path
		self.num_gpu = num_gpu
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.optimizer_name = optimizer_name
		self.lr = lr
		self.momentum = momentum
		self.stats_report_interval = stats_report_interval
		self.progress_check_interval = progress_check_interval

		# Initialize device
		if torch.cuda.is_available() and self.num_gpu > 0:
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")

		# Load model if necessary
		if load_model:
			print("Loading model...")
			self.model.load_state_dict(torch.load(self.model_path))
		
		# Move model to device and parallelize model if possible
		# @TODO: Try DistributedDataParallel?
		self.model = self.model.to(self.device)
		if self.device.type == "cuda" and self.num_gpu > 1:
			self.model = nn.DataParallel(self.model, list(range(self.num_gpu)))

		# Initialize optimizers for generator and discriminator
		self.D_optim = self.init_optim(self.model.D.parameters())
		self.G_optim = self.init_optim(self.model.G.parameters())

		# Initialize variables used for tracking loss and progress
		self.iters = 0
		self.D_losses = []
		self.G_losses = []
		self.fixed_sample_progress = []
		self.fixed_noise = torch.randn(64, self.model.num_latent, 1, 1, device=self.device)


	def init_optim(self, params):
		"""
		@TODO
		"""
		if self.optimizer_name == "adam":
			optim = torch.optim.Adam(params, lr=self.lr)
		elif self.optimizer_name == "sgd":
			optim = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
		elif self.optimizer_name == "rmsprop":
			optim = torch.optim.RMSprop(params, lr=self.lr, momentum=self.momentum)

		return optim


	def run(self, num_epochs=None, num_workers=0, save_results=False):
		"""
		Runs the trainer. Trainer will train the model then save it.
		Note that running trainer more than once will accumulate the results.

		Args:
			num_epochs: Run trainer for `num_epochs` epochs.
			num_workers: Number of worker loading the dataset.
			save_results: A flag indicating whether we should save the results this run.
		"""

		# This changes the number of epochs in trainer
		if num_epochs is not None:
			self.num_epochs = num_epochs

		# Initialize data loader
		data_loader = torch.utils.data.DataLoader(self.dataset,
			batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

		try:
			# Try training the model
			self.train(data_loader)
		finally:
			# Save model and report results
			print("Saving model...")
			torch.save(self.model.state_dict(), self.model_path)
			self.report_results(save_results)


	def train(self, data_loader):
		"""
		Trains `self.model` on data loaded from data loader.

		Args:
			data_loader: An iterator from which the data is sampled.
		"""

		print("Starting Training Loop...")
		# For each epoch
		for epoch in range(self.num_epochs):
			for batch_index, sample in enumerate(data_loader):

				# Sample real images
				real = sample["before"].to(self.device)

				# Calculate latent vector size
				batch_size = real.size()[0]
				latent_size = torch.Size([batch_size, self.model.num_latent, 1, 1])

				# Sample fake images from a random latent vector
				latent = torch.randn(latent_size, device=self.device)
				fake = self.model.G(latent)

				# Perform training step on discriminator and generator
				D_of_x, D_of_G_z1 = self.D_step(real, fake)
				D_of_G_z2 = self.G_step(fake)

				# Calculate index of data point
				index = batch_index * batch_size

				# Report training stats
				if batch_index % self.stats_report_interval == 0:
					self.report_training_stats(index, epoch, D_of_x, D_of_G_z1, D_of_G_z2)

				# Check generator's progress by recording its output on a fixed input
				if batch_index % self.progress_check_interval == 0:
					self.check_progress_of_generator(self.model.G)

				self.iters += 1

		# Show stats and check progress at the end
		self.report_training_stats(len(self.dataset), self.num_epochs, D_of_x, D_of_G_z1, D_of_G_z2)
		self.check_progress_of_generator(self.model.G)
		print("Finished training.")


	def D_step(self, real, fake):
		"""
		Trains the discriminator.

		Args:
			real: The real image sampled from the dataset.
			fake: The fake image generated by the generator.

		Returns:
			A tuple containing the mean classification of the discriminator on
			the real images and the fake images, respectively.
		"""

		# Zero gradients
		self.D_optim.zero_grad()

		# Initialize real and fake labels
		batch_size = real.size()[0]
		real_label = torch.full([batch_size], REAL, device=self.device)
		fake_label = torch.full([batch_size], FAKE, device=self.device)

		# Classify real images and calculate error
		D_on_real = self.model.D(real).view(-1)
		D_error_on_real = F.binary_cross_entropy(D_on_real, real_label)

		# Classify fake images and calculate error (don't pass gradients to G)
		D_on_fake = self.model.D(fake.detach()).view(-1)
		D_error_on_fake = F.binary_cross_entropy(D_on_fake, fake_label)

		# Calculate gradients from the error on real and fake images
		D_error = D_error_on_real + D_error_on_fake
		D_error.backward()
		
		# Update
		self.D_optim.step()

		# Record loss
		self.D_losses.append(D_error.item())

		return (
			D_on_real.mean().item(),
			D_on_fake.mean().item(),
		)


	def G_step(self, fake):
		"""
		Trains the generator.

		Args:
			fake: The fake image generated by the generator.

		Returns:
			The mean classification of the discriminator on the fake images.
		"""

		# Zero gradients
		self.G_optim.zero_grad()

		# Initialize fake labels
		batch_size = fake.size()[0]
		fake_label = torch.full([batch_size], FAKE, device=self.device)

		# Classify fake images and calculate generator's error
		# Note that we use 1-label because we want to maximize this step.
		# We can also back-propagate the -1*error, but this is more stable.
		D_on_fake = self.model.D(fake).view(-1)
		G_error = F.binary_cross_entropy(D_on_fake, 1 - fake_label)

		# Calculate gradients and update
		G_error.backward()
		self.G_optim.step()

		# Record loss
		self.G_losses.append(G_error.item())

		return D_on_fake.mean().item()



	def check_progress_of_generator(self, generator):
		"""
		Check generator's output on a fixed latent vector and record it.

		Args:
			generator: The generator which we want to check.
		"""

		with torch.no_grad():
			fixed_fake = generator(self.fixed_noise).detach().cpu()

		self.fixed_sample_progress.append(
			vutils.make_grid(fixed_fake, padding=2, normalize=True)
		)


	def report_training_stats(self, index, epoch, D_of_x, D_of_G_z1, D_of_G_z2, precision=4):
		"""
		Reports/prints the training stats to the console.

		Args:
			index: Index of the current data point.
			epoch: Current epoch.
			D_of_x: Mean classification of the discriminator on the real images.
			D_of_G_z1: Mean classification of the discriminator on the fake images (D_step).
			D_of_G_z2: Mean classification of the discriminator on the fake images (G_step).
			precision: Precision of the float numbers reported.
		"""

		report = \
			"[{epoch}/{num_epochs}][{index}/{len_dataset}]\t" \
			"Loss of D = {D_loss:.{p}f}\t" \
			"Loss of G = {G_loss:.{p}f}\t" \
			"D(x) = {D_of_x:.{p}f}\t" \
			"D(G(z)) = {D_of_G_z1:.{p}f} / {D_of_G_z2:.{p}f}"

		stats = {
			"epoch": epoch,
			"num_epochs": self.num_epochs,
			"index": index,
			"len_dataset": len(self.dataset),
			"D_loss": self.D_losses[-1],
			"G_loss": self.G_losses[-1],
			"D_of_x": D_of_x,
			"D_of_G_z1": D_of_G_z1,
			"D_of_G_z2": D_of_G_z2,
			"p": precision,
		}

		print(report.format(**stats))


	def report_results(self, save_results=False):
		"""
		Reports the result of the experiment.

		Args:
			save_results: Results will be saved if this was set to True.
		"""
		
		if save_results:
			# Get model directory and create experiment directory
			model_dir = os.path.dirname(self.model_path)
			experiment_dir = os.path.join(model_dir, self.get_experiment_name())
			if not os.path.isdir(experiment_dir): os.mkdir(experiment_dir)

			# Plot losses of D and G
			plot_file = os.path.join(experiment_dir, "losses.png")
			self.plot_losses(plot_file)

			# Create an animation of the generator's progress
			animation_file = os.path.join(experiment_dir, "progress.mp4")
			self.create_progress_animation(animation_file)
		else:
			self.plot_losses()
			self.create_progress_animation()


	def plot_losses(self, filename=None):
		"""
		Plots the losses of the discriminator and the generator.

		Args:
			filename: The plot's filename. if None, plot won't be saved.
		"""

		plt.figure(figsize=(10,5))
		plt.title("Generator and Discriminator Loss During Training")
		plt.plot(self.D_losses, label="D")
		plt.plot(self.G_losses, label="G")
		plt.xlabel("iterations")
		plt.ylabel("loss")
		plt.legend()
		if filename is not None: plt.savefig(filename)
		plt.show()


	def create_progress_animation(self, filename=None):
		"""
		Creates a video of the progress of the generator on a fixed latent vector.

		Args:
			filename: The animation's filename. if None, the animation won't be saved.
		"""

		fig = plt.figure(figsize=(8,8))
		plt.axis("off")
		ims = [[plt.imshow(img.permute(1,2,0), animated=True)]
			for img in self.fixed_sample_progress]
		ani = animation.ArtistAnimation(fig, ims, blit=True)
		
		if filename is not None: ani.save(filename)

		# Uncomment the following line to show this on a notebook
		# ani.to_jshtml()


	def get_experiment_name(self, delimiter=", "):
		"""
		Get the name of trainer's training train...

		Args:
			delimiter: The delimiter between experiment's parameters. Pretty useless.
		"""

		experiment_details = {
			"iters": self.iters,
			"batch_size": self.batch_size,
			"optim": self.optimizer_name,
			"lr": self.lr,
		}
		if self.optimizer_name != "adam":
			experiment_details["momentum"] = self.momentum

		timestamp = "[{}]".format(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
		experiment = delimiter.join("{}={}".format(k,v) for k,v in experiment_details.items())

		return timestamp + " " + experiment

