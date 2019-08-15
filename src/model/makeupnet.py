
import torch.nn as nn

from .gan import DCGAN


class MakeupNet(DCGAN):
    def __init__(self, **params):
        super().__init__(**params)



class MakeupApplier(nn.Module):
    """A Generator that applies makeup."""
    def __init__(self):
        super().__init__()


class MakeupRemover(nn.Module):
    """A Generator that removes makeup."""
    def __init__(self):
        super().__init__()


class DilatedResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()