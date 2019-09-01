
import torch.nn as nn

from torch.optim import SGD, RMSprop, Adam


def create_weights_init(conv_std=0.02, batchnorm_std=0.02):
    """
    A function that returns the weights initialization function for a net,
    which can be used as `net.apply(create_weights_init())`, for example.

    Args:
        conv_std: the standard deviation of the conv/up-conv layers.
        batchnorm_std: the standard deviation of the batch-norm layers.
    """

    def weights_init_apply(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, conv_std)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, batchnorm_std)
            nn.init.constant_(module.bias.data, 0)

    return weights_init_apply


def weights_init(module, conv_std=0.02, batchnorm_std=0.02):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, conv_std)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, batchnorm_std)
        nn.init.constant_(module.bias.data, 0)


def init_optim(params, optim_choice="sgd", lr=1e-4, momentum=0.0, betas=(0.9, 0.999)):
    """
    Initializes the optimizer.

    Args:
        params: Parameters the optimizer will optimize.
        choice: The choice of the optimizer.
        optim_configs: Configurations for the optimizer.

    Returns:
        The optimizer (torch.optim).
    """

    if optim_choice == "adam":
        optim = Adam(params, lr=lr, betas=betas)
    elif optim_choice == "rmsprop":
        optim = RMSprop(params, lr=lr)
    elif optim_choice == "sgd":
        optim = SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Optimizer '{optim_choice}' not recognized")

    return optim


    