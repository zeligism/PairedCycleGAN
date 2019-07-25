
import os
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize
from data.utility import files_iter

class MakeupDataset(Dataset):
    def __init__(self, dataset_dir, with_landmarks=False, transform=None):
        self.dataset_dir = dataset_dir
        self.with_landmarks = with_landmarks
        self.transform = transform
        self.images = self.get_images(self.dataset_dir)

    def with_landmarks(self):
        return self.with_landmarks

    def get_images(self, dataset_dir):
        all_images = list(files_iter(dataset_dir))
        before_images = list(filter(lambda s: s.find("before") != -1, all_images))
        after_images = list(filter(lambda s: s.find("after") != -1, all_images))

        return zip(sorted(before_images), sorted(after_images))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get path of before and after images
        (image_name_before, image_name_after) = self.images[index]
        path_before = os.path.join(self.root_dir, image_name_before)
        path_after  = os.path.join(self.root_dir, image_name_after)

        # Read images
        image_before = Image.open(path_before)
        image_after  = Image.open(path_after)

        landmarks = {}
        landmarks["before"] = self.load_landmarks(image_name_before),
        landmarks["after"] = self.load_landmarks(image_name_after),

        # Create sample
        sample = {
            "before": image_before,
            "after": image_after,
            "landmarks": landmarks,
        }

        # Apply transformations, if any
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def load_landmarks(self, image_name):
        
        landmarks = None

        if self.with_landmarks:
            # Get landmarks
            landmarks_name = image_name.split(".")[0] + ".pickle"
            landmarks_path = os.path.join(self.dataset_dir, "landmarks", landmarks_name)
            if os.path.exists(landmarks_path):
                with open(landmarks_path, "rb") as f:
                    landmarks = pickle.load(f)

        return landmarks


class ToTensor:

    def __call__(self, sample):

        # Get images
        img_before = sample["before"]
        img_after = sample["after"]

        # Resize output image
        img_size = img_before.size[::-1]  # im.size returns (w,h)
        img_after = resize(img_after, img_size)  # need (h,w) here

        # Transform PIL images to tensors
        img_before = to_tensor(img_before)
        img_after = to_tensor(img_after)

        # Transform landmarks to tensors
        landmarks = {}
        landmarks["before"] = self.landmarks_to_tensor(img_size, sample["landmarks"]["before"])
        landmarks["after"] = self.landmarks_to_tensor(img_size, sample["landmarks"]["after"])

        return {
            "before": img_before,
            "after": img_after,
            "landmarks": landmarks,
        }

    def landmarks_to_tensor(self, size, landmarks_dict):
        
        landmarks_tensor = None

        if landmarks_dict is not None:
            landmarks_indices = []
            for landmarks_part_indices in landmarks_dict:
                landmarks_indices += landmarks_part_indices

            landmarks_tensor = torch.zeros(size)
            landmarks_tensor[list(zip(*landmarks_indices))] = 1
            landmarks_tensor = landmarks_tensor / landmarks_tensor.norm()

        return landmarks_tensor

