
#import torch.nn as nn

from .gan import DCGAN

class MakeupNet(DCGAN):
    """The main module of the MakeupNet"""
    def __init__(self, **params):
        super().__init__(**params)



