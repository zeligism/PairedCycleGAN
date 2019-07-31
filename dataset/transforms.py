
import torch
from torchvision.transforms.functional import to_tensor, resize


class SampleTransform:
    """Transforms the samples from MakeupDataset."""

    def __init__(self, transform):
        """
        Initializes the transform.

        Args:
            transform: A transform (such as `torchvision.transforms.ToTensor()`).
        """
        self.transform = transform

    def __call__(self, sample):
        """
        Transforms the given sample.

        Args:
            sample: A sample from MakeupDataset to be transformed.

        Returns:
            The transform sample with `self.transform`.
        """

        transformed_sample = {
            "before": self.transform(sample["before"]),
            "after":  self.transform(sample["after"]),
        }

        if "landmarks" in sample:
            transformed_sample["landmarks"] = self.transform(sample["landmarks"])

        return transformed_sample


### Will Be Removed ###
class SampleToTensor:

    def __call__(self, sample):

        # Get images
        img_before = sample["before"]
        img_after = sample["after"]

        # Resize output image
        img_size = img_before.size[::-1]  # im.size returns (w,h)
        img_after = resize(img_after, img_size)  # need (h,w) here

        # Transform PIL images to tensors
        tensor_sample = {
            "before": to_tensor(img_before),
            "after":  to_tensor(img_after)
        }

        # Transform landmarks to tensors
        if "landmarks" in sample:
            landmarks = {
                "before": self.landmarks_to_tensor(img_size, sample["landmarks"]["before"]),
                "after":  self.landmarks_to_tensor(img_size, sample["landmarks"]["after"]),
            }
            tensor_sample["landmarks"] = landmarks


        return tensor_sample


    def landmarks_to_tensor(self, size, landmarks_dict):
        
        landmarks_tensor = torch.zeros(size)

        if landmarks_dict is not None:
            landmarks_indices = []
            for landmarks_part_indices in landmarks_dict:
                landmarks_indices += landmarks_part_indices

            landmarks_tensor[list(zip(*landmarks_indices))] = 1
            landmarks_tensor = landmarks_tensor / landmarks_tensor.norm()

        return landmarks_tensor


### Will Be Removed ###
class SampleResize:

    def __init__(self, size):
        self.size = size  # (height, width)


    def __call__(self, sample):
        
        resized_sample = {
            "before": resize(sample["before"], self.size),
            "after":  resize(sample["after"],  self.size),
        }

        if "landmarks" in sample:
            resized_sample["landmarks"] = sample["landmarks"]


        return resized_sample



