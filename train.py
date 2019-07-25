
import os
import time
import argparse
import torch
from torch.data.utils import DataLoader
from dataset.dataset import MakeupDataset, ToTensor


RANDOM_SEED = 1
CHANNEL_DIM = 0
HEIGHT_DIM = 1
WIDTH_DIM = 2

# Dataset parameters
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(FILE_DIR, "dataset", "data", "processing", "faces")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")

# Training parameters
OPTIMIZER = "sgd"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
CHECKPOINT = 1e4
SHOULD_REPORT = lambda i, k: (i + 1) % k == 0


def train(model, dataset,
	batch_size=4, lr=1e-4, optimizer="sgd", momentum=0.9, checkpoint=1e3):

	start_time = time.time()
	
	# Initialize device
	_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Create data loader
	dataset_loader = DataLoader(dataset,
		batch_size=batch_size, shuffle=True, num_workers=4)

	# Initialize optimizer
	if optimizer == "sgd":
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	elif optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	
	# Define loss function @TODO
	loss_function = lambda x, y: x - y


	# Define input and its true output
	for index, sample in enumerate(dataset_loader):

		before = sample["before"]
		after = sample["after"]
		landmarks = sample["landmarks"]

		if dataset.with_landmarks():
			before_mask = landmarks["before"] or torch.zeros(before.size())
			after_mask = landmarks["after"] or torch.zeros(after.size())
			torch.cat([before, before_mask], dim=CHANNEL_DIM)
			torch.cat([after, after_mask], dim=CHANNEL_DIM)

		# Do a forward pass, compute loss, then do a backward pass
		after_pred = model(before)
		loss = loss_function(after_pred, after)

		# Zero grads, calculate them, and update
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if SHOULD_REPORT(index, checkpoint):
			print("[{}/{}] Loss = {:.3f}".format(index+1, len(dataset), loss.mean().item()))
			print("Time elapsed = %ds" % (time.time() - start_time))


def main(args):
	# Set random seed if given
	torch.manual_seed(args.random_seed or torch.initial_seed())

	transform = ToTensor()
	dataset_params = {
		"dataset_dir": args.dataset_dir,
		"with_landmarks": args.with_landmarks,
		"transform": transform,
	}

	# Choose dataset and initialize size of data's input and output
	dataset = MakeupDataset(**dataset_params)

	# Initialize model @TODO
	model = ... # AutoMakeupNet(**model_params)

	training_params = {
		"batch_size": args.batch_size,
		"lr": args.learning_rate,
		"optimizer": args.optimizer,
		"momentum": args.momentum,
		"checkpoint": args.checkpoint,
	}
	train(model, dataset, **training_params)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="train AutoMakeupNet on MakeupDataset.")

	parser.add_argument('--random_seed', type=int, default=RANDOM_SEED,
		help="random seed (0 uses pytorch's initial seed).")

	parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
		help="directory of the makeup dataset.")
	parser.add_argument('--with_landmarks', action="store_true",
		help="use faces landmarks in training as well.")
	
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
		help="batch size.")
	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
		help="learning rate.")
	parser.add_argument('--optimizer', type=str.lower, default=OPTIMIZER, choices=("SGD", "Adam"),
		help="the optimizer used for training (SGD, Adam, etc.)")
	parser.add_argument('--momentum', type=float, default=MOMENTUM,
		help="the momentum used in SGD optimizer.")
	parser.add_argument('--checkpoint', type=float, default=CHECKPOINT,
		help="a report will be printed at every iteration that is divisible by the checkpoint index.")
	
	args = parser.parse_args()

	main(args)

