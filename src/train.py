
import os
import yaml
import argparse
import torch
import torchvision.transforms as transforms

import random
import numpy as np

from dataset.dataset import MakeupDataset
from dataset.transforms import MakeupSampleTransform
from model.makeupnet import _MakeupNet, MakeupNet
from trainers.makeupnet_trainer import _MakeupNetTrainer, MakeupNetTrainer
from trainers.utils.init_utils import create_weights_init

# @TODO: add logging

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(FILE_DIR, "dataset", "data", "processing", "faces")


def load_config(config_key, config_file="config.yaml"):
    """
    Load a configuration given by config_key from config_file.

    Args:
        config_key: Name/label/key of the configuration.
        config_file: Name of the config (yaml) file. Should be in current dir.

    Returns:
        The configurations as a dict.

    Throws:
        KeyError: if config_key is not found in the root level of the config.
    """
    with open(config_file) as f:
        all_configs = yaml.load(f)
        try:
            return all_configs[config_key]
        except KeyError:
            exit(f"Config key '{config_key}' is not defined in config file '{config_file}'.")


def get_dataset_args(args):
    """
    Construct dataset's parameters from args.

    Args:
        args: Parsed arguments from command line.
    """
    dataset_args = {
        "dataset_dir": args.dataset_dir,
        "with_landmarks": args.with_landmarks,
    }

    return dataset_args


def get_model_args(args):
    """
    Construct model's parameters from args.

    Args:
        args: Parsed arguments from command line.
    """
    model_args = {
        "num_latents": args.num_latents,
        "num_features": args.num_features,
        "image_channels": args.image_channels,
        "image_size": args.image_size,
        "gan_type": args.gan_type,
    }

    return model_args


def get_trainer_args(args):
    """
    Construct trainer's parameters from args.

    Args:
        args: Parsed arguments from command line.
    """
    trainer_args = {
        "name": args.trainer_name,
        "results_dir": args.results_dir,
        "load_model_path": args.load_model,
        "num_gpu": args.num_gpu,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,

        "D_optim_config": {
            "optim_choice": args.D_optimizer,
            "lr": args.D_lr,
            "momentum": args.D_momentum,
            "betas": args.D_betas,
        },
        "G_optim_config": {
            "optim_choice": args.G_optimizer,
            "lr": args.G_lr,
            "momentum": args.G_momentum,
            "betas": args.G_betas,
        },

        "D_iters": args.D_iters,
        "clamp": args.clamp,
        "gp_coeff": args.gp_coeff,
        "stats_interval": args.stats_interval,
        "generate_grid_interval": args.generate_grid_interval,
    }

    return trainer_args


def set_random_seed(seed):
    """
    Sets all the random seeds to `seed`.

    Args:
        seed: Initial random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    """
    Trains the MakeupNet on MakeupDataset using MakeupNetTrainer.

    Args:
        args: The arguments passed from the command prompt (see below for more info).
    """

    set_random_seed(args.random_seed)

    # Initialize args for dataset, model, and trainer
    if args.config is not None:
        config = load_config(args.config)
        dataset_args = config["dataset"]
        model_args = config["model"]
        trainer_args = config["trainer"]
    else:
        dataset_args = get_dataset_args(args)
        model_args = get_model_args(args)
        trainer_args = get_trainer_args(args)

    # Define data transformation
    image_size = model_args["image_size"]
    transform_sequence = list(map(MakeupSampleTransform, [
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    transform = transforms.Compose(transform_sequence)

    # Define weights initialization method
    weights_init = create_weights_init()

    # Initialize dataset, model, and trainer
    dataset = MakeupDataset(**dataset_args, transform=transform)
    model = MakeupNet(**model_args)
    trainer = MakeupNetTrainer(model, dataset, **trainer_args, weights_init=weights_init)

    # Train model on dataset using trainer
    trainer.run(num_epochs=args.num_epochs, save_results=args.save_results)


if __name__ == "__main__":

    # Adding positive and non-negative types for arguments type check
    def positive(type):    
        def positive_type(value):
            typed_value = type(value)
            if not (typed_value > 0):
                raise argparse.ArgumentTypeError(f"{value} is not a positive {type.__name__}.")
            return typed_value
        return positive_type

    def nonnegative(type):    
        def nonnegative_type(value):
            typed_value = type(value)
            if not (typed_value >= 0):
                raise argparse.ArgumentTypeError(f"{value} is not a non-negative {type.__name__}.")
            return typed_value
        return nonnegative_type


    # Initialize parser and add arguments
    parser = argparse.ArgumentParser(description="train MakeupNet on MakeupDataset using MakeupNetTrainer.")

    parser.add_argument("--config", type=str,
        help="The key of the configurations of dataset, model, and trainer as defined in 'config.yaml'. "
             "This will override all given args for dataset, model, and trainer.")

    parser.add_argument("--random_seed", type=int, default=123,
        help="random seed.")

    ### Dataset Args ###
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR,
        help="directory of the makeup dataset.")
    parser.add_argument("--with_landmarks", action="store_true",
        help="use faces landmarks in training as well.")

    ### Model Args ###
    parser.add_argument("--num_latents", type=positive(int), default=128,
        help="number of latent factors from which an image will be generated.")
    parser.add_argument("--num_features", type=positive(int), default=64,
        help="number of features on the layers of the discriminator (and the generator as well).")
    parser.add_argument("--image_channels", type=positive(int), default=3,
        help="number of image channels in the dataset.")
    parser.add_argument("--image_size", type=positive(int), default=64,
        help="resize images to be of dimensions (image_size x image_size).")
    parser.add_argument("--gan_type", type=str.lower, default="gan",
        choices=("gan", "wgan", "wgan-gp"),
        help="type of gan among GAN (default), WGAN (Wasserstein GAN), and WGAN-GP (WGAN with gradient penalty).")

    ### Trainer Args ###
    parser.add_argument("--trainer_name", type=str, default="trainer",
        help="name of the model trainer (which is also the name of your experiment).")
    parser.add_argument("--results_dir", type=str, default="results/",
        help="directory where the results for each run will be saved.")
    parser.add_argument("--load_model", type=str,
        help="the path of the file where the model will be loaded and experiments will be saved.")
    
    parser.add_argument("--num_gpu", type=nonnegative(int), default=0,
        help="number of GPUs to use, if any.")
    parser.add_argument("--num_workers", type=nonnegative(int), default=0,
        help="number of workers that will be loading the dataset.")
    parser.add_argument("--batch_size", type=positive(int), default=4,
        help="size of the batch sample from the dataset.")

    parser.add_argument("--D_optimizer", type=str.lower, default="sgd",
        help="the name of the optimizer used for training (SGD, Adam, RMSProp)",
        choices=("sgd", "adam", "rmsprop"),)
    parser.add_argument("--D_lr", type=float, default=1.0e-4,
        help="the learning rate, which controls the size of the optimization update.")
    parser.add_argument("--D_momentum", type=positive(float), default=0.0,
        help="used in SGD and RMSProp optimizers.")
    parser.add_argument("--D_betas", type=float, nargs=2, default=(0.9, 0.999),
        help="used in Adam optimizer (see torch.optim.Adam for details).")

    parser.add_argument("--G_optimizer", type=str.lower, default="sgd",
        help="the name of the optimizer used for training (SGD, Adam, RMSProp)",
        choices=("sgd", "adam", "rmsprop"),)
    parser.add_argument("--G_lr", type=float, default=1.0e-4,
        help="the learning rate, which controls the size of the optimization update.")
    parser.add_argument("--G_momentum", type=positive(float), default=0.0,
        help="used in SGD and RMSProp optimizers.")
    parser.add_argument("--G_betas", type=float, nargs=2, default=(0.9, 0.999),
        help="used in Adam optimizer (see torch.optim.Adam for details).")

    parser.add_argument("--D_iters", type=positive(int), default=5,
        help="number of iterations to train discriminator every batch.")
    parser.add_argument("--clamp", type=float, nargs=2, default=(-0.01, 0.01),
        help="used in WGAN for clamping the weights of the discriminator.")
    parser.add_argument("--gp_coeff", type=float, default=10.0,
        help="a coefficient to multiply with the gradient penalty in the loss of WGAN-GP.")

    parser.add_argument("--stats_interval", type=positive(int), default=50,
        help="the interval in which a report of the training stats will be shown to the console.")
    parser.add_argument("--generate_grid_interval", type=positive(int), default=200,
        help="the interval in which the progress of the generator will be checked and recorded.")

    ### Trainer.run() ###
    parser.add_argument("--num_epochs", type=positive(int), default=5,
        help="number of training epochs (i.e. full runs on the dataset).")
    parser.add_argument("--save_results", action="store_true",
        help="save the results of the experiment.")
    
    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args)

