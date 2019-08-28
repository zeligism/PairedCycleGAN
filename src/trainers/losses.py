
import torch
import torch.nn.functional as F

# @TODO: docs

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


def D_grad_penalty(D_grad_norm, gp_coeff):

    # D's gradient penalty is `gp_coeff * (|| grad of D(x_i) wrt x_i || - 1)^2`
    grad_penalty = (D_grad_norm - 1).pow(2) * gp_coeff

    return grad_penalty


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



