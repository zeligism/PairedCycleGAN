
import os
import datetime
import torch
import torch.utils.tensorboard as tensorboard

from pprint import pformat
from collections import defaultdict
from .utils.report_utils import plot_lines


class BaseTrainer:
    """The base trainer class."""

    def __init__(self, model, dataset,
        name="trainer",
        results_dir="results/",
        load_model_path=None,
        num_gpu=1,
        num_workers=0,
        batch_size=4,
        report_interval=10,
        save_interval=100000,
        use_tensorboard=False,  # XXX: not implemented yet
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
            report_interval: Report stats every `report_interval` iters.
            save_interval: Save model every `save_interval` iters.
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

        self.report_interval = report_interval
        self.save_interval = save_interval
        self.description = description
        self.save_results = False

        self.start_time = datetime.datetime.now()
        self.stop_time = datetime.datetime.now()
        self.iters = 1  # current iteration (i.e. # of batches processed so far)
        self.batch = 1  # current batch
        self.epoch = 1  # current epoch
        self.num_batches = 1 + len(self.dataset) // self.batch_size  # num of batches per epoch
        self.num_epochs = 0  # number of epochs to run

        self._dataset_sampler = iter(())  # generates samples from the dataset
        self._data = defaultdict(list)  # contains data of experiment

        self.writer = None
        self.use_tensorboard = use_tensorboard

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


    def time_since_start(self):
        elapsed_time = datetime.datetime.now() - self.start_time
        return elapsed_time.total_seconds()


    def run(self, num_epochs, save_results=False):
        """
        Runs the trainer. Trainer will train the model and then save it.
        Note that running trainer more than once will accumulate the results.

        Args:
            num_epochs: Number of epochs to run.
            save_results: A flag indicating whether we should save the results this run.
        """
        self.start_time = datetime.datetime.now()
        self.num_epochs = num_epochs + self.epoch - 1
        self.save_results = save_results

        # Create experiment directory
        experiment_name = self.get_experiment_name()
        experiment_dir = os.path.join(self.results_dir, experiment_name)
        if self.save_results:
            if not os.path.isdir(self.results_dir): os.mkdir(self.results_dir)
            if not os.path.isdir(experiment_dir): os.mkdir(experiment_dir)

        with tensorboard.SummaryWriter(f"runs/{experiment_name}") as self.writer:
            # Try training the model, then stop the training when an exception is thrown
            try:
                self.train()
            finally:
                self.stop_time = datetime.datetime.now()
                self.stop()


    def train(self):
        """
        Train model on dataset for `num_epochs` epochs.

        Args:
            num_epochs: Number of epochs to run.
        """

        # Train until dataset sampler is exhausted (i.e. until it throws StopIteration)
        self.init_dataset_sampler()

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


    def init_dataset_sampler(self):
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
        self._dataset_sampler = iter(self.sample_loader(loader_config))


    def sample_loader(self, loader_config):
        """
        A generator that yields samples from the dataset, exhausting it `num_epochs` times.

        Args:
            num_epochs: Number of epochs.
            loader_config: Configuration for pytorch's data loader.
        """

        for self.epoch in range(self.epoch, self.num_epochs + 1):
            data_loader = torch.utils.data.DataLoader(self.dataset, **loader_config)
            for self.batch, sample in enumerate(data_loader, 1):
                yield sample

        self.epoch += 1


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
        should_report_stats = self.iters % self.report_interval == 0
        should_save_progress = self.iters % self.save_interval == 0
        finished_epoch = self.batch == self.num_batches

        # Report training stats
        if should_report_stats or finished_epoch:
            self.report_stats()

        if self.save_results and should_save_progress:
            model_path = os.path.join(self.results_dir,
                                      self.get_experiment_name(),
                                      f"model@{self.iters}.pt")
            self.save_model(model_path)


    def stop(self):
        """
        Stops the trainer, or what happens when the trainer stops.
        Note: This will run even on keyboard interrupts.
        """

        # plot losses, if any
        plot_lines(self.get_data_containing("loss"), title="Losses")


    def get_experiment_name(self, delimiter=", "):
        """
        Get the name of trainer's training train...

        Args:
            delimiter: The delimiter between experiment's parameters. Pretty useless.
        """
        info = {
            "name": self.name,
            "batch_size": self.batch_size,
        }

        timestamp = self.start_time.strftime("%y%m%d-%H%M%S")
        experiment = delimiter.join(f"{k}={v}" for k,v in info.items())

        return "[{}] {}".format(timestamp, experiment)


    def report_stats(self, precision=3):
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

