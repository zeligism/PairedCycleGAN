
import os
import torch
import torch.utils.data as data_utils

import random
import glob
from PIL import Image
from face_recognition import face_landmarks


def dict_to_list(d):
    return [x for l in d.values() for x in l]


def files_iter(directory):
    return filter(os.path.isfile, glob.iglob(os.path.join(directory, "*")))


class MakeupDataset(data_utils.Dataset):
    """A dataset of before-and-after makeup images."""

    def __init__(self, dataset_dir,
                 transform=None,
                 with_landmarks=False,
                 paired=False,
                 reverse=False):
        """
        Initializes MakeupDataset.

        Args:
            dataset_dir: The directory of the dataset.
            transform: The transform used on the data.
            with_landmarks: A flag indicating whether landmarks should be used or not.
            paired: Indicates whether images should be paired when sampled or not.
            reverse: Reverses sample if True (before = with makeup, after = no makeup).
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.with_landmarks = with_landmarks
        self.paired = paired
        self.reverse = reverse

        self.images_before, self.images_after = self.get_images()
        self.landmarks_cache = {}
        self.landmarks_size = [72, 2]


    def get_images(self):
        """
        Return a list of pairs of (before, after) makeup images name in `dataset_dir`.

        Returns:
            A list of tuples of the names of before and after makeup images in `dataset_dir`.
        """
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")

        all_images = list(files_iter(self.dataset_dir))
        before_images = list(filter(lambda s: s.find("before") != -1, all_images))
        after_images = list(filter(lambda s: s.find("after") != -1, all_images))

        return sorted(before_images), sorted(after_images)


    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.images_before)


    def __getitem__(self, index):
        """
        Get the next data point from the dataset.

        Args:
            index: the index of the data point.

        Returns:
            The data point transformed and ready for consumption.
        """

        # Shuffle the other list/dataset every time we reiterate from the beginning
        if not self.paired and index == 0:
            random.shuffle(self.images_after)

        # Sample before and after images
        image_before = self.images_before[index]
        image_after = self.images_after[index]

        # Create sample
        sample = {
            "before": Image.open(image_before).convert("RGB"),
            "after":  Image.open(image_after).convert("RGB"),
        }

        # Apply transformations on images
        if self.transform is not None:
            sample = self.transform(sample)

        # Find landmarks, use cache if already done
        if self.with_landmarks:
            sample["landmarks"] = {
                "before": self.get_landmarks(image_before, sample["before"]),
                "after":  self.get_landmarks(image_after,  sample["after"]),
            }

        # Reverse direction of sample
        if self.reverse:
            sample = self.reverse_sample(sample)

        return sample


    def get_landmarks(self, label, image):
        """
        Get the landmarks associated with the label and image.
        If label is not in landmarks' cache, find the landmarks in image.

        Args:
            label: The label of the image.
            image: Image in PyTorch tensor format.

        Returns:
            Landmarks in PyTorch tensor format.
        """

        if label in self.landmarks_cache:
            landmarks = self.landmarks_cache[label]
        else:
            landmarks = self.find_landmarks(image)
            self.landmarks_cache[label] = landmarks
        
        return landmarks


    def find_landmarks(self, image):
        """
        Find the landmarks of an image.

        Args:
            image: image in PyTorch tensor format.

        Returns:
            The landmarks of the image as a tensor.
        """

        unnormalize = lambda t: t * 0.5 + 0.5  # XXX: hard-coded un-normalization
        to_uint8_rgb = lambda t: (255 * t).round().to(torch.uint8)
        torch_to_numpy = lambda t: t.permute(1, 2, 0).numpy()

        # Image is a pytorch tensor, prepare it as an image in standard numpy format
        image = torch_to_numpy(to_uint8_rgb(unnormalize(image)))

        # Find landmarks in the image
        landmarks_found = face_landmarks(image)
        # If found any, return first one as a tensor, else return zeros
        if len(landmarks_found) > 0:
            landmarks = torch.tensor(dict_to_list(landmarks_found[0]), dtype=torch.int)
        else:
            landmarks = torch.zeros(self.landmarks_size, dtype=torch.int)

        return landmarks


    def reverse_sample(self, sample):
        """
        Reverse direction of sample.

        Args:
            sample: A sample from the dataset
        """

        sample = {
            "before": sample["after"],
            "after": sample["before"],
        }

        if "landmarks" in sample:
            sample["landmarks"] = {
                "before": sample["landmarks"]["after"],
                "after": sample["landmarks"]["before"],
            }


        return sample


    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.dataset_dir)



class MakeupDataset2(MakeupDataset):
    """A new, expanded, unpaired version of MakeupDataset."""

    def __init__(self, dataset_dir,
                 transform=None,
                 with_landmarks=False,
                 reverse=False):
        # Initialize as an unpaired MakeupDataset
        super().__init__(dataset_dir, transform=transform,
            with_landmarks=with_landmarks, paired=False, reverse=reverse)

    def get_images(self):
        """
        Return a list of pairs of (before, after) makeup images name in `dataset_dir`.

        Returns:
            A list of tuples of the names of before and after makeup images in `dataset_dir`.
        """
        nomakeup_dir = os.path.join(self.dataset_dir, "nomakeup")
        makeup_dir = os.path.join(self.dataset_dir, "makeup")

        if not os.path.isdir(nomakeup_dir):
            raise FileNotFoundError(f"No-Makeup directory '{nomakeup_dir}' does not exist.")
        if not os.path.isdir(makeup_dir):
            raise FileNotFoundError(f"Makeup directory '{makeup_dir}' does not exist.")

        before_images = list(files_iter(nomakeup_dir))
        after_images = list(files_iter(makeup_dir))

        return sorted(before_images), sorted(after_images)


