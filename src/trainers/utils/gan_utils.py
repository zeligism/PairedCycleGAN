
import torch
import torch.nn.functional as F

# @TODO: Docs.

def D_step(D_optim, D, real, fake,
           gan_type="gan", clamp=(-0.1, 0.1), gp_coeff=10.):
    """
    Trains the discriminator D and maximizes its score function.

    Args:
        D_optim: The optimizer for D.
        D: The discriminator.
        real: Real data point sampled from the dataset.
        fake: Fake data point sampled from the generator.

    Returns:
        A tuple containing the loss, the mean classification of D on
        the real images, as well as on the fake images, respectively.
    """

    # Zero gradients
    D_optim.zero_grad()

    # Classify real and fake images
    D_on_real = D(real)
    D_on_fake = D(fake)

    if gan_type == "gan":
        D_loss = D_loss_GAN(D_on_real, D_on_fake)

    elif gan_type == "wgan":
        D_loss = D_loss_WGAN(D_on_real, D_on_fake)

    elif gan_type == "wgan-gp":
        D_grad_norm = get_D_grad_norm(D, real, fake)
        grad_penalty = get_grad_penalty(D_grad_norm, gp_coeff)
        D_loss = D_loss_WGAN(D_on_real, D_on_fake, grad_penalty=grad_penalty)

    else:
        raise ValueError(f"gan_type {gan_type} not supported")

    # Calculate gradients and minimize loss
    D_loss.backward()
    D_optim.step()

    # If WGAN, clamp D's weights to ensure k-Lipschitzness
    if gan_type == "wgan":
        [p.data.clamp_(*clamp) for p in D.parameters()]

    return {
        "D_loss": D_loss.item(),
        "D_on_real": D_on_real.mean().item(),
        "D_on_fake": D_on_fake.mean().item(),
    }


def G_step(G_optim, D, fake, gan_type="gan"):
    """
    Trains the generator and minimizes its score function.

    Args:
        G_optim: The optimizer for G.
        D: The discriminator.
        fake: Fake data point generated from a latent variable.

    Returns:
        The mean loss of D on the fake images as well as
        the mean classification of the discriminator on the fake images.
    """

    # Zero gradients
    G_optim.zero_grad()

    # Classify fake images
    D_on_fake = D(fake)

    if gan_type == "gan":
        G_loss = G_loss_GAN(D_on_fake)

    elif gan_type == "wgan" or gan_type == "wgan-gp":
        G_loss = G_loss_WGAN(D_on_fake)

    else:
        raise ValueError(f"gan_type {gan_type} not supported")

    # Calculate gradients and minimize loss
    G_loss.backward()
    G_optim.step()

    return {
        "G_loss": G_loss.item(),
        "D_on_fake": D_on_fake.mean().item()
    }


def get_D_loss(D, real, fake, gan_type="gan", gp_coeff=10.):

    # Classify real and fake images
    D_on_real = D(real)
    D_on_fake = D(fake)

    if gan_type == "gan":
        loss = D_loss_GAN(D_on_real, D_on_fake)

    elif gan_type == "wgan":
        loss = D_loss_WGAN(D_on_real, D_on_fake)

    elif gan_type == "wgan-gp":
        D_grad_norm = get_D_grad_norm(D, real, fake)
        grad_penalty = get_grad_penalty(D_grad_norm, gp_coeff)
        loss = D_loss_WGAN(D_on_real, D_on_fake, grad_penalty=grad_penalty)

    else:
        raise ValueError(f"gan_type {gan_type} not supported")

    return loss


def get_G_loss(D, fake, gan_type="gan"):

    # Classify fake images
    D_on_fake = D(fake)

    if gan_type == "gan":
        loss = G_loss_GAN(D_on_fake)

    elif gan_type == "wgan" or gan_type == "wgan-gp":
        loss = G_loss_WGAN(D_on_fake)

    else:
        raise ValueError(f"gan_type {gan_type} not supported")

    return loss


def D_loss_GAN(D_on_real, D_on_fake):
    
    # Create (noisy) real and fake labels
    real_label = 0.8 + 0.2 * torch.rand_like(D_on_real)
    fake_label = 0.05 * torch.rand_like(D_on_fake)
    
    # Calculate binary cross entropy loss
    D_loss_on_real = F.binary_cross_entropy(D_on_real, real_label)
    D_loss_on_fake = F.binary_cross_entropy(D_on_fake, fake_label)
    
    # Loss is: - log(D(x)) - log(1 - D(x_g)),
    # which is equiv. to maximizing: log(D(x)) + log(1 - D(x_g))
    D_loss = torch.mean(D_loss_on_real + D_loss_on_fake)

    return D_loss


def D_loss_WGAN(D_on_real, D_on_fake, grad_penalty=0.0):

    # Maximize: D(x) - D(x_g) - gp_coeff * (|| grad of D(x_i) wrt x_i || - 1)^2,
    # where x_i <- eps * x + (1 - eps) * x_g, and eps ~ rand(0,1)
    D_loss = -1 * torch.mean(D_on_real - D_on_fake - grad_penalty)

    return D_loss


def G_loss_GAN(D_on_fake):

    # Calculate binary cross entropy loss with a fake binary label
    fake_label = torch.zeros_like(D_on_fake)

    # Loss is: -log(D(G(z))), which is equiv. to minimizing log(1-D(G(z)))
    # We use this loss vs. the original one for stability only.
    G_loss = F.binary_cross_entropy(D_on_fake, 1 - fake_label)

    return G_loss


def G_loss_WGAN(D_on_fake):

    # Minimize: -D(G(z))
    G_loss = (-D_on_fake).mean()
    
    return G_loss


def get_D_grad_norm(discriminator, real, fake):

    batch_size = real.size()[0]
    device = real.device

    # Calculate gradient penalty
    eps = torch.rand([batch_size, 1, 1, 1], device=device)
    interpolated = eps * real + (1 - eps) * fake
    interpolated.requires_grad_()
    D_on_inter = discriminator(interpolated)

    # Calculate gradient of D(x_i) wrt x_i for each batch
    D_grad = torch.autograd.grad(D_on_inter, interpolated,
                                 torch.ones_like(D_on_inter), retain_graph=True)

    # D_grad will be a 1-tuple, as in: (grad,)
    D_grad_norm = D_grad[0].view([batch_size, -1]).norm(dim=1)

    return D_grad_norm


def get_grad_penalty(grad_norm, gp_coeff=10.):

    # D's gradient penalty is `gp_coeff * (|| grad of D(x_i) wrt x_i || - 1)^2`
    grad_penalty = (grad_norm - 1).pow(2) * gp_coeff

    return grad_penalty


