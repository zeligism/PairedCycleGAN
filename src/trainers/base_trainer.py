
import os
import datetime
import torch

from pprint import pformat
from collections import defaultdict
from .utils.report_utils import plot_lines


# @TODO: implement TrainerRecord
# @TODO: implement TrainerSchedule


class BaseTrainer:
    """The base trainer class."""

    def __init__(self, model, dataset,
        name="trainer",
        results_dir="results/",
        load_model_path=None,
        num_gpu=1,
        num_workers=0,
        batch_size=4,
        stats_interval=10,
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
            stats_interval: Report stats every `stats_interval` batch.
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

        self.stats_interval = stats_interval
        self.description = description

        self.iters = 1  # current iteration (i.e. # of batches processed so far)
        self.batch = 1  # current batch
        self.epoch = 1  # current epoch
        self.num_batches = 1 + len(self.dataset) // self.batch_size  # num of batches per epoch
        self.num_epochs = 0  # number of epochs to run

        self._dataset_sampler = iter(())  # generates samples from the dataset
        self._data = defaultdict(list)  # contains data of experiment

        # Load model if necessary
        if load_model_path is not None:
            self.load_model(load_model_path)
        
        # Initialize device
        using_cuda = torch.cuda.is_available() and self.num_gpu > 0
        self.device = torch.device("cuda:0" if using_cuda else "cpu")

        # Move model to device and parallelize model if possible
        self.model = self.model.to(self.device)
        if self.device.type == "cuda" and self.num_gpu > 1:
            self.model = torch.nn.DistributedDataParallel(self.model, list(range(self.num_gpu)))


    def load_model(self, model_path):
        if not os.path.isfile(model_path):
            print(f"Couldn't load model: file '{model_path}' does not exist")
            print("Training model from scratch.")
        else:
            print("Loading model...")
            self.model.load_state_dict(torch.load(model_path))


    def save_model(self, model_path):
        print("Saving model...")
        torch.save(self.model.state_dict(), model_path)


    def run(self, num_epochs, save_results=False):
        """
        Runs the trainer. Trainer will train the model and then save it.
        Note that running trainer more than once will accumulate the results.

        Args:
            num_epochs: Number of epochs to run.
            save_results: A flag indicating whether we should save the results this run.
        """

        if save_results:
            # Create results directory if it hasn't been created yet
            if not os.path.isdir(self.results_dir): os.mkdir(self.results_dir)

        try:
            # Try training the model
            self.train(num_epochs)
        finally:
            # Always stop trainer and report results
            self.stop(save_results)


    def train(self, num_epochs):
        """
        Train model on dataset for `num_epochs` epochs.

        Args:
            num_epochs: Number of epochs to run.
        """

        # Train until dataset sampler is exhausted (i.e. until it throws StopIteration)
        self.init_dataset_sampler(num_epochs)

        try:
            print(f"Starting training {self.name}...")
            while True:
                # One training step/iteration
                self.pre_train_step()
                self.train_step()
                self.post_train_step()
                self.iters += 1

        except StopIteration:
            print("Finished training.")


    def init_dataset_sampler(self, num_epochs):
        """
        Initializes the sampler (or iterator) of the dataset.

        Args:
            num_epochs: Number of epochs.
        """
        loader_config = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
        }
        self._dataset_sampler = iter(self.sample_loader(num_epochs, loader_config))


    def sample_loader(self, num_epochs, loader_config):
        """
        A generator that yields samples from the dataset, exhausting it `num_epochs` times.

        Args:
            num_epochs: Number of epochs.
            loader_config: Configuration for pytorch's data loader.
        """
        
        self.num_epochs = num_epochs + self.epoch - 1  # last epoch

        for self.epoch in range(self.epoch, self.num_epochs + 1):
            data_loader = torch.utils.data.DataLoader(self.dataset, **loader_config)
            for self.batch, sample in enumerate(data_loader, 1):
                yield sample


    def sample_dataset(self):
        """
        Samples the dataset. To be called by the client.

        Returns:
            A sample from the dataset.
        """
        return next(self._dataset_sampler)


    def pre_train_step(self):
        """
        The training preparation, or what happens before each training step.
        """
        pass


    def train_step(self):
        """
        Makes one training step.
        """
        pass


    def post_train_step(self):
        """
        The training checkpoint, or what happens after each training step.
        """
        # Report training stats
        if self.batch % self.stats_interval == 0:
            self.report_training_stats()


    def stop(self, save_results=False):
        """
        Stops the trainer, or what happens when the trainer stops.
        Note: This will run even on keyboard interrupts.

        Args:
            save_results: Results will be saved if True.
        """
        plot_lines(self.get_data_containing("loss"), title="Losses")


    def get_experiment_name(self, delimiter=", "):
        """
        Get the name of trainer's training train...

        Args:
            delimiter: The delimiter between experiment's parameters. Pretty useless.
        """
        info = {
            "name": self.name,
            "iters": self.iters - 1,
            "batch_size": self.batch_size,
        }

        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        experiment = delimiter.join(f"{k}={v}" for k,v in info.items())

        return "[{}] {}".format(timestamp, experiment)


    def report_training_stats(self, precision=3):
        """
        Default training stats report.
        Prints the current value of each data list recorded.
        """

        # Progress of training
        progress = f"[{self.epoch}/{self.num_epochs}][{self.batch}/{self.num_batches}]  "

        # Show the stat of an item
        item_stat = lambda item: f"{item[0]} = {item[1][-1]:.{precision}f}"
        # Join the stats separated by tabs
        stats = ",  ".join(map(item_stat, self._data.items()))

        report = progress + stats

        print(report)


    def get_current_value(self, label):
        """
        Get the current value of the quantity given by `label`.

        Args:
            label: Name/label of the data/quantity.

        Returns:
            The current value of the quantity given by `label`.
        """
        return self._data[label][-1] if len(self._data[label]) > 0 else None


    def get_data_containing(self, phrase):
        """
        Get the data lists that contain `phrase` in their names/labels.

        Args:
            phrase: A phrase to find in the label of the data, such as "loss".

        Returns:
            A dict containing the data lists that contain `phrase` in their labels.
        """
        return {k: v for k, v in self._data.items() if k.find(phrase) != -1}


    def add_data(self, **kwargs):
        """
        Adds/appends a value to the list given by `label`.

        Args:
            kwargs: Dict of values to be added to data lists corresponding to their labels.
        """
        for key, value in kwargs.items():
            self._data[key].append(value)


    def __repr__(self):

        self_dict = dict({k:v for k,v in self.__dict__.items() if k[0] != "_"})
        pretty_dict = pformat(self_dict)
        
        return self.__class__.__name__ + "(**" + pretty_dict + ")"


