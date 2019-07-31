
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

from .data.utility import files_iter


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
        self.dataset_dir = dataset_dir
        self.with_landmarks = with_landmarks
        self.transform = transform
        self.images = self.get_images(self.dataset_dir)


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

        # Create and add landmarks if needed
        if self.with_landmarks:
            landmarks = {
                "before": self.load_landmarks(image_name_before),
                "after":  self.load_landmarks(image_name_after),
            }
            sample["landmarks"] = landmarks

        # Apply transformations, if any
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def get_images(self, dataset_dir):
        """
        Return a list of pairs of (before, after) makeup images name in `dataset_dir`.

        Args:
            dataset_dir: The directory of the dataset.

        Returns:
            A list of tuples of the names of before and after makeup images in `dataset_dir`.
        """

        all_images = list(files_iter(dataset_dir))
        before_images = list(filter(lambda s: s.find("before") != -1, all_images))
        after_images = list(filter(lambda s: s.find("after") != -1, all_images))

        return list(zip(sorted(before_images), sorted(after_images)))


    def load_landmarks(self, image_name):
        """
        Load the landmarks associated with `image_name`.

        Args:
            image_name: The name of the image of which we want the landmarks.

        Returns:
            The landmarks associated with `image_name`.
        """

        landmarks = None

        # Get landmarks
        landmarks_name = image_name.split(".")[0] + ".pickle"
        landmarks_path = os.path.join(self.dataset_dir, "landmarks", landmarks_name)
        if os.path.exists(landmarks_path):
            with open(landmarks_path, "rb") as f:
                landmarks = pickle.load(f)

        return self.landmarks_to_pil_image(landmarks)


    def landmarks_to_pil_image(self, landmarks):
        """@TODO"""
        return landmarks


