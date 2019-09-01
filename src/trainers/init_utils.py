
import torch.nn as nn


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


def init_optim(self, optim_name, params, optim_configs):
    """
    Initializes the optimizer.

    Args:
        optim_name: The choice of the optimizer.
        params: Parameters the optimizer will optimize.
        optim_configs: Configurations for the optimizer.

    Returns:
        The optimizer (torch.optim).
    """

    optimizers = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
    }
    
    if optim_name not in optimizers:
        raise ValueError(f"Optimizer '{optim_name}' not recognized.")

    optimizer = optimizers[optim_name]

    return optimizer(params, **optim_configs)


    