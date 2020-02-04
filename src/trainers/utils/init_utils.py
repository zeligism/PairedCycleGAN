
import torch
import torch.nn as nn


def create_weights_init(conv_std=0.02, batchnorm_std=0.02):
    """
    A function that returns the weights initialization function for a net,
    which can be used as `net.apply(create_weights_init())`, for example.

    Args:
        conv_std: the standard deviation of the conv/up-conv layers.
        batchnorm_std: the standard deviation of the batch-norm layers.
    """

    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, conv_std)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, batchnorm_std)
            nn.init.constant_(module.bias.data, 0)

    def weights_init_kaiming(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    return weights_init_kaiming


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
        optim = torch.optim.Adam(params, lr=lr, betas=betas)
    elif optim_choice == "rmsprop":
        optim = torch.optim.RMSprop(params, lr=lr)
    elif optim_choice == "sgd":
        optim = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Optimizer '{optim_choice}' not recognized.")

    return optim


    