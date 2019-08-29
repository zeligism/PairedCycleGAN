
import os
import argparse
import torch
import torchvision.transforms as transforms

from dataset.dataset import MakeupDataset
from dataset.transforms import MakeupSampleTransform
from model.makeupnet import MakeupNet
from trainers.makeupnet_trainer import MakeupNetTrainer

# @TODO: add logging
# @TODO: put hyperparams in yaml

### DEFAULT HYPERPARAMETERS ###

# Random seed
RANDOM_SEED = 123

### Dataset parameters ###
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(FILE_DIR, "dataset", "data", "processing", "faces")
IMAGE_SIZE = 64

### Network parameters ###
GAN_TYPE = "gan"
NUM_CHANNELS = 3
NUM_LATENTS = 128
NUM_FEATURES = 64

### Training parameters ###
TRAINER_NAME = "trainer"
RESULTS_DIR = "results/"
LOAD_MODEL_PATH = None

NUM_GPU = 1
NUM_WORKERS = 2
BATCH_SIZE = 4

# Optimizer's hyperparameters are dependent on the type of gan.
# Make sure you change these accordingly.
# @TODO: Add separate hyperparams for D and G?

OPTIMIZER_NAME = "sgd"
LEARNING_RATE = 1e-4
MOMENTUM = 0.
BETAS = (0.9, 0.999)

if GAN_TYPE == "sgd":
    OPTIMIZER_NAME = "adam"
    LEARNING_RATE = 1e-4

elif GAN_TYPE == "wgan":
    OPTIMIZER_NAME = "rmsprop" # or adam
    LEARNING_RATE = 5e-5
    BETAS = (0.5, 0.9)

elif GAN_TYPE == "wgan-gp":
    OPTIMIZER_NAME = "adam"
    LEARNING_RATE = 1e-4
    BETAS = (0.0, 0.9)

D_ITERS = 5
CLAMP = (-0.01, 0.01)  # for WGAN
GRADIENT_PENALTY_COEFF = 10.  # for WGAN-GP

# Checkpoints frequency
STATS_REPORT_INTERVAL = 50
PROGRESS_CHECK_INTERVAL = 200

NUM_EPOCHS = 5

### END ###


def main(args):
    """
    Trains the MakeupNet on MakeupDataset using MakeupNetTrainer.

    Args:
        args: The arguments passed from the command prompt (see below for more info).
    """

    # Unpack args and pass them to training objects
    torch.manual_seed(args.random_seed or torch.initial_seed())

    # Define data transformations
    transform_list = list(map(MakeupSampleTransform, [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    # Define dataset parameters
    dataset_params = {
        "dataset_dir": args.dataset_dir,
        "with_landmarks": args.with_landmarks,
        "transform": transforms.Compose(transform_list),
    }

    # Define model parameters
    model_params = {
        "num_channels": args.num_channels,
        "num_latents": args.num_latents,
        "num_features": args.num_features,
        "image_size": args.image_size,
        "gan_type": args.gan_type,
    }

    # Define training parameters
    trainer_params = {
        "name": args.trainer_name,
        "results_dir": args.results_dir,
        "load_model_path": args.load_model,
        "num_gpu": args.num_gpu,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "optimizer_name": args.optimizer,
        "lr": args.learning_rate,
        "momentum": args.momentum,
        "betas": tuple(args.betas),
        "D_iters": args.D_iters,
        "clamp": tuple(args.clamp),
        "gp_coeff": args.gp_coeff,
        "stats_report_interval": args.stats_interval,
        "progress_check_interval": args.progress_interval,
    }

    # Start initializing dataset, model, and trainer
    dataset = MakeupDataset(**dataset_params)
    model = MakeupNet(**model_params)
    trainer = MakeupNetTrainer(model, dataset, **trainer_params)

    # Train MakeupNet
    trainer.run(num_epochs=args.num_epochs, save_results=args.save_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train MakeupNet on MakeupDataset.")

    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
        help="random seed (0 uses pytorch initial seed).")

    ### Dataset ###
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR,
        help="directory of the makeup dataset.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
        help="resize images to be of dimensions (image_size x image_size).")
    parser.add_argument("--with_landmarks", action="store_true",
        help="use faces landmarks in training as well.")
    
    ### Model ###
    parser.add_argument("--gan_type", type=str.lower, default=GAN_TYPE,
        choices=("gan", "wgan", "wgan-gp"),
        help="type of gan among GAN (default), WGAN (Wasserstein GAN), and WGAN-GP (WGAN with gradient penalty).")
    parser.add_argument("--num_channels", type=int, default=NUM_CHANNELS,
        help="number of image channels in the dataset.")
    parser.add_argument("--num_latents", type=int, default=NUM_LATENTS,
        help="number of latent factors from which an image will be generated.")
    parser.add_argument("--num_features", type=int, default=NUM_FEATURES,
        help="number of features on the layers of the discriminator (and the generator as well).")

    ### Trainer ###
    parser.add_argument("--trainer_name", type=str, default=TRAINER_NAME,
        help="name of the model trainer (which is also the name of your experiment).")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
        help="directory where the results for each run will be saved.")
    parser.add_argument("--load_model", type=str, default=LOAD_MODEL_PATH,
        help="the path of the file where the model will be loaded and experiments will be saved.")
    
    parser.add_argument("--num_gpu", type=int, default=NUM_GPU,
        help="number of GPUs to use, if any.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
        help="number of workers that will be loading the dataset.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
        help="size of the batch sample from the dataset.")

    parser.add_argument("--optimizer", type=str.lower, default=OPTIMIZER_NAME,
        help="the name of the optimizer used for training (SGD, Adam, RMSProp)",
        choices=("sgd", "adam", "rmsprop"),)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
        help="the learning rate, which controls the size of the optimization update.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
        help="used in SGD and RMSProp optimizers.")
    parser.add_argument("--betas", type=float, nargs=2, default=BETAS,
        help="used in Adam optimizer (see torch.optim.Adam for details).")

    parser.add_argument("--D_iters", type=int, default=D_ITERS,
        help="number of iterations to train discriminator every batch.")
    parser.add_argument("--clamp", type=float, nargs=2, default=CLAMP,
        help="used in WGAN for clamping the weights of the discriminator.")
    parser.add_argument("--gp_coeff", type=float, default=GRADIENT_PENALTY_COEFF,
        help="a coefficient to multiply with the gradient penalty in the loss of WGAN-GP.")

    parser.add_argument("--stats_interval", type=int, default=STATS_REPORT_INTERVAL,
        help="the interval in which a report of the training stats will be shown to the console.")
    parser.add_argument("--progress_interval", type=int, default=PROGRESS_CHECK_INTERVAL,
        help="the interval in which the progress of the generator will be checked and recorded.")

    ### Trainer.run() ###
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
        help="number of training epochs (i.e. full runs on the dataset).")
    parser.add_argument("--save_results", action="store_true",
        help="save the results of the experiment.")
    
    # Parse arguments
    args = parser.parse_args()

    # Additional validation of input @TODO: create positive_int nonnegative_int instead
    if not (args.image_size > 0):
        parser.error("Image size should be a positive integer.")

    if not (args.num_channels > 0):
        parser.error("Number of channels should be a positive integer.")
    if not (args.num_latents > 0):
        parser.error("Number of latent dimensions should be a positive integer.")
    if not (args.num_features > 0):
        parser.error("Number of features per layer should be a positive integer.")

    if not (args.num_gpu >= 0):
        parser.error("Number of GPUs should be a non-negative integer.")
    if not (args.num_workers >= 0):
        parser.error("Number of data loading workers should be a non-negative integer.")
    if not (args.batch_size > 0):
        parser.error("Batch size should be a positive integer.")

    if not (args.D_iters > 0):
        parser.error("Number of discriminator iterations should be a positive integer.")
    if not (args.momentum >= 0):
        parser.error("Momentum should be a positive integer.")
    for beta in args.betas:
        if not (0. <= beta and beta < 1.):
            parser.error("Betas should be numbers in the range [0, 1).")

    if not (args.stats_interval > 0):
        parser.error("Stats report interval should be a positive integer.")
    if not (args.progress_interval > 0):
        parser.error("Progress check interval should be a positive integer.")

    if not (args.num_epochs > 0):
        parser.error("Number of epochs should be a positive integer.")

    # Run main function

    main(args)

