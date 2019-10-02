
import os
import torch
import numpy as np

from PIL import Image
from face_recognition import face_landmarks
from torch.utils.data import Dataset
from .data.utility import files_iter

dict_to_list = lambda d: [x for l in d.values() for x in l]


class MakeupDataset(Dataset):
    """A dataset of before-and-after makeup images."""

    def __init__(self, dataset_dir, with_landmarks=False, transform=None):
        """
        Initializes MakeupDataset.

        Args:
            dataset_dir: The directory of the dataset.
            with_landmarks: A flag indicating whether landmarks should be used or not.
            transform: The transform used on the data.
        """

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

        self.dataset_dir = dataset_dir
        self.with_landmarks = with_landmarks
        self.transform = transform
        self.images = self.get_images()

        self.landmarks_cache = [None] * len(self.images)
        self.landmarks_size = [72, 2]


    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.images)


    def __getitem__(self, index):
        """
        Get the next data point from the dataset.

        Args:
            index: the index of the data point.

        Returns:
            The data point transformed and ready for consumption.
        """

        # Get path of before and after images
        (image_name_before, image_name_after) = self.images[index]
        path_before = os.path.join(self.dataset_dir, image_name_before)
        path_after  = os.path.join(self.dataset_dir, image_name_after)

        # Read images, convert to RGB
        image_before = Image.open(path_before).convert("RGB")
        image_after  = Image.open(path_after).convert("RGB")

        # Create sample
        sample = {
            "before": image_before,
            "after":  image_after
        }

        # Apply transformations on images
        if self.transform is not None:
            sample = self.transform(sample)

        # Find landmarks, use cache if already done
        if self.with_landmarks:
            if self.landmarks_cache[index] is None:
                landmarks = self.find_landmarks(sample)
                self.landmarks_cache[index] = landmarks
            else:
                landmarks = self.landmarks_cache[index]

            sample["landmarks"] = landmarks

        return sample


    def get_images(self):
        """
        Return a list of pairs of (before, after) makeup images name in `dataset_dir`.

        Returns:
            A list of tuples of the names of before and after makeup images in `dataset_dir`.
        """

        all_images = list(files_iter(self.dataset_dir))
        before_images = list(filter(lambda s: s.find("before") != -1, all_images))
        after_images = list(filter(lambda s: s.find("after") != -1, all_images))

        return list(zip(sorted(before_images), sorted(after_images)))


    def find_landmarks(self, sample):
        """
        Find the landmarks of the images in the sample.

        Args:
            sample: A sample from the dataset.

        Returns:
            The landmarks associated with the sample.
        """

        unnormalize = lambda t: t * 0.5 + 0.5  # @XXX: hard-coded un-normalization
        to_uint8_rgb = lambda t: (255 * t).round().to(torch.uint8)
        torch_to_numpy = lambda t: t.permute(1, 2, 0).numpy()

        landmarks = {}

        for label, image in sample.items():
            # Image is a pytorch tensor, prepare it as an image in standard numpy format
            image = torch_to_numpy(to_uint8_rgb(unnormalize(image)))
            landmarks_found = face_landmarks(image)
            if len(landmarks_found) > 0:
                landmarks[label] = torch.tensor(dict_to_list(landmarks_found[0]), dtype=torch.int)
            else:
                landmarks[label] = torch.zeros(self.landmarks_size, dtype=torch.int)

        return landmarks


    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.dataset_dir)


