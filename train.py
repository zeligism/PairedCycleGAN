
import os
import argparse
import torch
import torchvision.transforms as transforms

from dataset.dataset import MakeupDataset
from dataset.transforms import SampleTransform
from model.makeupnet import MakeupNet
from trainer import MakeupNetTrainer


### DEFAULT PARAMETERS ###

# Random seed
RANDOM_SEED = 123

### Dataset parameters ###
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(FILE_DIR, "dataset", "data", "processing", "faces")

### Network parameters ###
NUM_CHANNELS = 3
NUM_LATENT = 100
NUM_FEATURES = 64

### Training parameters ###
MODEL_PATH = "model/makeupnet.pt"
NUM_GPU = 1
NUM_EPOCHS = 5
BATCH_SIZE = 4
OPTIMIZER_NAME = "sgd"
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
CHECKPOINT = 1e4

### END ###


def main(args):
	"""
	Trains the MakeupNet on MakeupDataset using MakeupNetTrainer.

	Args:
		args: The arguments passed from the command prompt (see below for more info).
	"""

	torch.manual_seed(args.random_seed or torch.initial_seed())

	# Define data transformations
	transform = transforms.Compose([
		SampleTransform(transforms.Resize((64, 64))),
		SampleTransform(transforms.ToTensor()),
	])

	# Define dataset parameters
	dataset_params = {
		"dataset_dir": args.dataset_dir,
		"with_landmarks": args.with_landmarks,
		"transform": transform,
	}

	# Define model parameters
	model_params = {
		"num_channels": args.num_channels,
		"num_latent": args.num_latent,
		"num_features": args.num_features,
	}

	# Define training parameters
	training_params = {
		"load_model": args.load_model,
		"model_path": args.model_path,
		"num_gpu": args.num_gpu,
		"num_epochs": args.num_epochs,
		"batch_size": args.batch_size,
		"optimizer_name": args.optimizer_name,
		"lr": args.learning_rate,
		"momentum": args.momentum,
		#"checkpoint": args.checkpoint,
	}

	# Start initializing dataset, model, and trainer
	dataset = MakeupDataset(**dataset_params)
	model = MakeupNet(**model_params)
	trainer = MakeupNetTrainer(model, dataset, **training_params)

	# Train MakeupNet
	trainer.start()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="train MakeupNet on MakeupDataset.")

	parser.add_argument("-rd", '--random_seed', type=int, default=RANDOM_SEED,
		help="random seed (0 uses pytorch's initial seed).")

	parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
		help="directory of the makeup dataset.")
	parser.add_argument('--with_landmarks', action="store_true",
		help="use faces landmarks in training as well.")
	
	parser.add_argument('--num_channels', type=int, default=NUM_CHANNELS,
		help="number of image channels in the dataset.")
	parser.add_argument('--num_latent', type=int, default=NUM_LATENT,
		help="number of latent factors from which an image will be generated.")
	parser.add_argument('--num_features', type=int, default=NUM_FEATURES,
		help="number of features on the layers of the discriminator (and the generator as well).")

	parser.add_argument('--load_model', action="store_true",
		help="load model from the given (or default) model file.")
	parser.add_argument('--model_path', type=str, default=MODEL_PATH,
		help="the path of the file where the model will be loaded and saved.")
	parser.add_argument('--num_gpu', type=int, default=NUM_GPU,
		help="number of GPUs to use, if any.")
	parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
		help="number of training epochs (i.e. full runs on the dataset).")
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
		help="batch size.")
	parser.add_argument('--optimizer_name', type=str.lower, default=OPTIMIZER_NAME,
		choices=("sgd", "adam", "rmsprop"),
		help="the name of the optimizer used for training (SGD, Adam, etc.)")
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
		help="the learning rate used in the optimizer.")
	parser.add_argument('--momentum', type=float, default=MOMENTUM,
		help="the momentum used in the optimizer, if applicable.")
	parser.add_argument('--checkpoint', type=float, default=CHECKPOINT,
		help="a report will be printed at every iteration that is divisible by the checkpoint index.")
	
	args = parser.parse_args()

	main(args)

