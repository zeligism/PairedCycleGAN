
import os
import yaml
import argparse
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from dataset.dataset import MakeupDataset
from dataset.transforms import MakeupSampleTransform

from models.cyclegan import MaskCycleGAN
from models.pairedcyclegan import PairedCycleGAN

from trainers.cyclegan_trainer import CycleGAN_Trainer
from trainers.pairedcyclegan_trainer import PairedCycleGAN_Trainer
from trainers.utils.init_utils import create_weights_init


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(FILE_DIR, "dataset", "data", "processing", "faces")


def parse_args():
    """
    Parse training args.
    """

    # Adding positive and non-negative types for arguments type check
    def positive(type):    
        def positive_number(value):
            typed_value = type(value)
            if not (typed_value > 0):
                raise argparse.ArgumentTypeError(f"{value} is not a positive {type.__name__}.")
            return typed_value
        return positive_number

    def nonnegative(type):    
        def nonnegative_number(value):
            typed_value = type(value)
            if not (typed_value >= 0):
                raise argparse.ArgumentTypeError(f"{value} is not a non-negative {type.__name__}.")
            return typed_value
        return nonnegative_number


    # Initialize parser and add arguments
    parser = argparse.ArgumentParser(description="Train a model on a dataset using a trainer.")

    parser.add_argument("-c", "--config", type=str,
        help="The key of the configurations of dataset, model, and trainer as defined in 'config.yaml'. "
             "This will override all given args for dataset, model, and trainer.")

    parser.add_argument("-r", "--random-seed", type=int, default=123,
        help="random seed.")

    ### Dataset Args ###
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
        help="directory of the makeup dataset.")

    ### Model Args ###
    parser.add_argument("--num-latents", type=positive(int), default=128,
        help="number of latent factors from which an image will be generated.")
    parser.add_argument("--num-features", type=positive(int), default=64,
        help="number of features on the layers of the discriminator (and the generator as well).")
    parser.add_argument("--image-channels", type=positive(int), default=3,
        help="number of image channels in the dataset.")
    parser.add_argument("--image-size", type=positive(int), default=64,
        help="resize images to be of dimensions (image_size x image_size).")
    parser.add_argument("--gan-type", type=str.lower, default="gan",
        choices=("gan", "wgan", "wgan-gp"),
        help="type of gan among GAN (default), WGAN (Wasserstein GAN), and WGAN-GP (WGAN with gradient penalty).")

    ### Trainer Args ###
    parser.add_argument("--results-dir", type=str, default="results/",
        help="directory where the results for each run will be saved.")
    parser.add_argument("--load-model", type=str,
        help="the path of the file where the model will be loaded and experiments will be saved.")
    
    parser.add_argument("--num-gpu", type=nonnegative(int), default=0,
        help="number of GPUs to use, if any.")
    parser.add_argument("--num-workers", type=nonnegative(int), default=0,
        help="number of workers that will be loading the dataset.")
    parser.add_argument("--batch-size", type=positive(int), default=4,
        help="size of the batch sample from the dataset.")

    parser.add_argument("--D-optimizer", type=str.lower, default="sgd",
        help="the name of the optimizer used for training (SGD, Adam, RMSProp)",
        choices=("sgd", "adam", "rmsprop"),)
    parser.add_argument("--D-lr", type=float, default=1.0e-4,
        help="the learning rate, which controls the size of the optimization update.")
    parser.add_argument("--D-momentum", type=positive(float), default=0.0,
        help="used in SGD and RMSProp optimizers.")
    parser.add_argument("--D-betas", type=float, nargs=2, default=(0.9, 0.999),
        help="used in Adam optimizer (see torch.optim.Adam for details).")

    parser.add_argument("--G-optimizer", type=str.lower, default="sgd",
        help="the name of the optimizer used for training (SGD, Adam, RMSProp)",
        choices=("sgd", "adam", "rmsprop"),)
    parser.add_argument("--G-lr", type=float, default=1.0e-4,
        help="the learning rate, which controls the size of the optimization update.")
    parser.add_argument("--G-momentum", type=positive(float), default=0.0,
        help="used in SGD and RMSProp optimizers.")
    parser.add_argument("--G-betas", type=float, nargs=2, default=(0.9, 0.999),
        help="used in Adam optimizer (see torch.optim.Adam for details).")

    parser.add_argument("--D-iters", type=positive(int), default=5,
        help="number of iterations to train discriminator every batch.")
    parser.add_argument("--clamp", type=float, nargs=2, default=(-0.01, 0.01),
        help="used in WGAN for clamping the weights of the discriminator.")
    parser.add_argument("--gp-coeff", type=float, default=10.0,
        help="a coefficient to multiply with the gradient penalty in the loss of WGAN-GP.")

    parser.add_argument("--report-interval", type=positive(int), default=50,
        help="the interval in which a report of the training stats will be shown to the console.")
    parser.add_argument("--save-interval", type=positive(int), default=10000,
        help="the interval in which the model will be saved.")
    parser.add_argument("--generate-grid-interval", type=positive(int), default=200,
        help="the interval in which the progress of the generator will be checked and recorded.")

    ### Trainer.run() ###
    parser.add_argument("-e", "--num-epochs", type=positive(int), default=5,
        help="number of training epochs (i.e. full runs on the dataset).")
    parser.add_argument("-s", "--save-results", action="store_true",
        help="save the results of the experiment.")

    # Parse arguments
    args = parser.parse_args()

    return args


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
        all_configs = yaml.full_load(f)
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
        "report_interval": args.report_interval,
        "save_interval": args.save_interval,
        "generate_grid_interval": args.generate_grid_interval,
    }

    return trainer_args


def set_random_seed(seed):
    """
    Sets all random seeds to `seed`.

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


def get_training_args(args):
    """
    Return structured args for dataset, model, and trainer from parsed args.
    """

    if args.config is not None:
        config = load_config(args.config)
        dataset_args = config["dataset"]
        model_args = config["model"]
        trainer_args = config["trainer"]
    else:
        dataset_args = get_dataset_args(args)
        model_args = get_model_args(args)
        trainer_args = get_trainer_args(args)

    return dataset_args, model_args, trainer_args


def make_transform(image_size):
    """
    Make data transform and return it.
    """
    transform_sequence = [
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.01),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform_sequence = list(map(MakeupSampleTransform, transform_sequence))
    transform = transforms.Compose(transform_sequence)

    return transform


def main(args):
    """
    Trains the MakeupNet on MakeupDataset using MakeupNetTrainer.

    Args:
        args: The arguments passed from the command prompt (see below for more info).
    """

    set_random_seed(args.random_seed)

    # Initialize args for dataset, model, and trainer
    dataset_args, model_args, trainer_args = get_training_args(args)

    # Define data transformation and weights initializer
    transform = make_transform(model_args["image_size"])
    weights_init = create_weights_init()


    # Train makeup remover using CycleGAN
    unpaired_dataset = MakeupDataset(**dataset_args, transform=transform, reverse=True)
    
    facecleaner_cyclegan = MaskCycleGAN(**model_args)
    facecleaner_cyclegan.apply(weights_init)

    subtrainer = CycleGAN_Trainer(facecleaner_cyclegan, unpaired_dataset,
                                  name="makeupgan.remover", **trainer_args)
    subtrainer.run(num_epochs=args.num_epochs, save_results=args.save_results)


    # Train PairedCycleGAN, and assign to it the pre-trained makeup remover
    paired_dataset = MakeupDataset(**dataset_args, transform=transform,
                                   with_landmarks=True, paired=True)

    makeup_pcgan = PairedCycleGAN(**model_args)
    makeup_pcgan.remover = facecleaner_cyclegan.applier  # as in "applying the makeup cleaning"
    makeup_pcgan.applier.apply(weights_init)

    trainer = PairedCycleGAN_Trainer(makeup_pcgan, paired_dataset,
                                     name="makeupgan", **trainer_args)
    trainer.run(num_epochs=args.num_epochs, save_results=args.save_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)

