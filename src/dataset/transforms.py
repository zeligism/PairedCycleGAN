

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

        sample["before"] = self.transform(sample["before"])
        sample["after"] = self.transform(sample["after"])

        return sample

