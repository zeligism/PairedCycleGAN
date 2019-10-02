
import os
import datetime
import torch

from pprint import pformat
from .utils.init_utils import weights_init


class BaseTrainer:
    """The base trainer class."""

    def __init__(self, model, dataset,
        name="trainer",
        results_dir="results/",
        load_model_path=None,
        num_gpu=1,
        num_workers=2,
        batch_size=4,
        description="no description given",
        **kwargs):
        """
        Initializes BaseTrainer.

        Args:
            model: The model or net.
            dataset: The dataset on which the model will be training.
            name: Name of this trainer.
            results_dir: Directory in which results will be saved for each run.
            load_model_path: Path to the model that will be loaded, if any.
            num_gpu: Number of GPUs to use for training.
            num_workers: Number of workers sampling from the dataset.
            batch_size: Size of the batch. Must be > num_gpu.
            description: Description of the experiment the trainer is running.
        """

        self.model = model
        self.dataset = dataset

        self.name = name
        self.results_dir = results_dir
        self.load_model_path = load_model_path

        self.num_gpu = num_gpu
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.description = description

        self.iters = 1  # current iteration

        # Load model if necessary
        if load_model_path is not None:
            self.load_model(load_model_path)
        
        # Initialize device
        using_cuda = torch.cuda.is_available() and num_gpu > 0
        self.device = torch.device("cuda:0" if using_cuda else "cpu")

        # Move model to device and parallelize model if possible
        self.model = self.model.to(self.device)
        if self.device.type == "cuda" and self.num_gpu > 1:
            self.model = torch.nn.DistributedDataParallel(self.model, list(range(self.num_gpu)))

        # Initialize model
        self.model.apply(weights_init)


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

                # Do things before training step (check progress, record data, etc)
                self.pre_train_step(epoch, num_epochs, batch, num_batches)

                # Train model on the sample
                self.train_step(sample)

                # Do things after training step (check progress, record data, etc)
                self.post_train_step(epoch, num_epochs, batch, num_batches)

                self.iters += 1

        # Do a post-training step at the end as well
        self.post_train_step(num_batches, num_batches, num_epochs, num_epochs)

        print("Finished training.")


    def train_step(self, sample):
        """
        Trains on a sample from the dataset.

        Args:
            sample: Real data points sampled from the dataset.
        """
        raise NotImplementedError("train_step() should be implemented!")


    def pre_train_step(self, epoch, num_epochs, batch, num_batches):
        """
        The training preparation, or what happens before each training step.

        Args:
            epoch: Current epoch.
            num_epochs: Number of epochs to run.
            batch: Current batch.
            num_batches: Number of batches to run.
        """
        pass


    def post_train_step(self, epoch, num_epochs, batch, num_batches):
        """
        The training checkpoint, or what happens after each training step.

        Args:
            epoch: Current epoch.
            num_epochs: Number of epochs to run.
            batch: Current batch.
            num_batches: Number of batches to run.
        """
        pass


    def stop(self, save_results=False):
        """
        Stops the trainer, or what happens when the trainer stops.
        Note: This will run even on keyboard interrupts.

        Args:
            save_results: Results will be saved if True.
        """
        pass


    def get_experiment_name(self, delimiter=", "):
        """
        Get the name of trainer's training train...

        Args:
            delimiter: The delimiter between experiment's parameters. Pretty useless.
        """
        info = {
            "name": self.name,
            "iters": self.iters - 1,
        }

        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        experiment = delimiter.join(f"{k}={v}" for k,v in info.items())

        return "[{}] {}".format(timestamp, experiment)
        

    def __repr__(self):

        self_dict = dict({k:v for k,v in self.__dict__.items() if k[0] != "_"})
        pretty_dict = pformat(self_dict)
        
        return self.__class__.__name__ + "(**" + pretty_dict + ")"


