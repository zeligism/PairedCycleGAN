
import torch
import torch.nn.functional as F


def get_D_loss(gan_type="gan"):
    if gan_type in ("gan", "gan-gp"):
        return D_loss_GAN
    elif gan_type in ("wgan", "wgan-gp"):
        return D_loss_WGAN
    else:
        raise ValueError(f"gan_type {gan_type} not supported")


def get_G_loss(gan_type="gan"):
    if gan_type in ("gan", "gan-gp"):
        return G_loss_GAN
    elif gan_type in ("wgan", "wgan-gp"):
        return G_loss_WGAN
    else:
        raise ValueError(f"gan_type {gan_type} not supported")


def D_loss_GAN(D_on_real, D_on_fake, label_smoothing=True):
    
    # Create (noisy) real and fake labels XXX
    if label_smoothing:
        real_label = 0.7 + 0.5 * torch.rand_like(D_on_real)
    else:
        real_label = torch.ones_like(D_on_real) - 0.1
    fake_label = torch.zeros_like(D_on_fake)

    # Calculate binary cross entropy loss
    D_loss_on_real = F.binary_cross_entropy(D_on_real, real_label)
    D_loss_on_fake = F.binary_cross_entropy(D_on_fake, fake_label)

    # Loss is: - log(D(x)) - log(1 - D(x_g)),
    # which is equiv. to maximizing: log(D(x)) + log(1 - D(x_g))
    D_loss = D_loss_on_real + D_loss_on_fake

    return D_loss.mean()


def D_loss_WGAN(D_on_real, D_on_fake, grad_penalty=0.0):

    # Maximize: D(x) - D(x_g) - const * (|| grad of D(x_i) wrt x_i || - 1)^2,
    # where x_i <- eps * x + (1 - eps) * x_g, and eps ~ rand(0,1)
    D_loss = -1 * (D_on_real - D_on_fake - grad_penalty)

    return D_loss.mean()


def G_loss_GAN(D_on_fake):

    # Calculate binary cross entropy loss with a fake binary label
    fake_label = torch.zeros_like(D_on_fake)

    # Loss is: -log(D(G(z))), which is equiv. to minimizing log(1-D(G(z)))
    # We use this loss vs. the original one for stability only.
    G_loss = F.binary_cross_entropy(D_on_fake, 1 - fake_label)

    return G_loss.mean()


def G_loss_WGAN(D_on_fake):

    # Minimize: -D(G(z))
    G_loss = -D_on_fake
    
    return G_loss.mean()


def random_interpolate(real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1).to(real)
    return eps * real + (1 - eps) * fake


def simple_gradient_penalty(D, x, center=0.):
    x.requires_grad_()
    D_on_x = D(x)
    D_grad = torch.autograd.grad(D_on_x, x, torch.ones_like(D_on_x), create_graph=True)
    D_grad_norm = D_grad[0].view(x.size(0), -1).norm(dim=1)
    return (D_grad_norm - center).pow(2).mean()
