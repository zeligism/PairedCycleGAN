
from torchvision.transforms.functional import to_tensor, resize


class MakeupSampleTransform:
    """A wrapper around torch transforms for
    transforming samples from MakeupDataset."""

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
            transformed_sample["landmarks"] = {
            "before": self.transform(sample["landmarks"]["before"]),
            "after": self.transform(sample["landmarks"]["after"]),
        }

        return transformed_sample

