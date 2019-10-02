
import torch
import torch.nn.functional as F

from .gan_trainer import GAN_Trainer
from .base_trainer import BaseTrainer
from .utils.init_utils import init_optim
from .utils.gan_utils import *
from .utils.face_morph.face_morph import face_morph


class _MakeupNetTrainer(GAN_Trainer):
    """The trainer for MakeupNet."""

    def __init__(self, model, dataset, **kwargs):
        """
        Initializes MakeupNetTrainer.

        Args:
            model: The makeup net.
            dataset: The makeup dataset.
        """
        super().__init__(model, dataset, **kwargs)


class MakeupNetTrainer(BaseTrainer):
    """The trainer for MakeupNet."""

    def __init__(self, model, dataset,
                 D_optim_config={},
                 G_optim_config={},
                 D_iters=5,
                 clamp=(-0.01, 0.01),
                 gp_coeff=10.,
                 stats_interval=50,
                 **kwargs):
        """
        Initializes MakeupNetTrainer.

        Args:
            model: The makeup net.
            dataset: The makeup dataset.
        """
        super().__init__(model, dataset, **kwargs)

        self.D_iters = D_iters
        self.clamp = clamp
        self.gp_coeff = gp_coeff
        self.stats_interval = stats_interval

        # Initialize optimizers for generator and discriminator
        self.optims = {
            "applier": {
                "D": init_optim(self.model.applier.D.parameters(), **D_optim_config),
                "G": init_optim(self.model.applier.G.parameters(), **G_optim_config),
            },
            "remover": {
                "D": init_optim(self.model.remover.D.parameters(), **D_optim_config),
                "G": init_optim(self.model.remover.G.parameters(), **G_optim_config),
            },
            "style": {
                "D": init_optim(self.model.style_D.parameters(), **D_optim_config),
                # "G": init_optim([]),
            }
        }

        # Initialize data dict
        nan = torch.tensor(float("nan"))
        self._data = {
            "D_loss": [nan],
            "G_loss": [nan],
        }


    def optims_zero_grad(self, D_or_G):
        for optim in self.optims:
            if D_or_G in optim:
                optim[D_or_G].zero_grad()


    def optims_step(self, D_or_G):
        for optim in self.optims:
            if D_or_G in optim:
                optim[D_or_G].step()


    def train_step(self, sample):

        print("Step: %d" % self.iters)

        makeup_applier = self.model.applier
        makeup_remover = self.model.remover
        makeup_style_D = self.model.style_D
        
        ### Sampling ###

        # Sample from dataset
        real_makeup = sample["after"].to(self.device)
        real_nomakeup = sample["before"].to(self.device)
        real_makeup_lm = sample["landmarks"]["after"]
        real_nomakeup_lm = sample["landmarks"]["before"]

        # Sample from generators @TODO: requires_grad?
        with torch.no_grad():
            fake_makeup = makeup_applier.G(real_nomakeup, real_makeup).detach()
            fake_nomakeup = makeup_remover.G(real_makeup).detach()

        # Morph makeup face to nomakeup face's facial structure for style loss calculation
        mask, morphed_real_makeup = self.morph_makeup(real_makeup, real_nomakeup,
                                                      real_makeup_lm, real_nomakeup_lm)
        # Prepare real same style pair vs. fake same style pair
        real_style_pair = (mask * real_makeup, mask * morphed_real_makeup)
        fake_style_pair = (mask * real_makeup, mask * fake_makeup)

        ### D step ###

        # Zero gradients and loss
        self.optims_zero_grad("D")

        # Initialize D's loss
        D_loss = 0.

        # Adversarial loss for makeup domain
        D_loss += 0.1 * get_D_loss(makeup_applier.D, real_makeup, fake_makeup,
                                   gan_type=self.model.gan_type, gp_coeff=self.gp_coeff)

        # Adversarial loss for no-makeup domain
        D_loss += 0.1 * get_D_loss(makeup_remover.D, real_nomakeup, fake_nomakeup,
                                   gan_type=self.model.gan_type, gp_coeff=self.gp_coeff)

        # "Adversarial" loss for style domain (@XXX: gan_type shouldn't involve grad norm)
        D_loss += 0.1 * get_D_loss(makeup_style_D, real_style_pair, fake_style_pair,
                                   gan_type="gan", gp_coeff=self.gp_coeff)
        
        # Calculate gradients
        D_loss.backward()

        # Make a step of minimizing D's loss
        self.optims_step("D")

        ### End of D step ###

        self._data["D_loss"].append(D_loss)


        # Train generator if we trained discriminator D_iters time
        if self.iters % self.D_iters == 0:

            ### Sampling ###
            
            # Sample from generators
            fake_makeup = makeup_applier.G(real_nomakeup, real_makeup)
            fake_nomakeup = makeup_remover.G(real_makeup)


            ### G step ###

            # Zero gradients
            self.optims_zero_grad("G")

            # Initialize G's loss
            G_loss = 0.

            # Adversarial loss for makeup domain
            G_loss += 0.1 * get_G_loss(makeup_applier.D, fake_makeup, gan_type=self.model.gan_type)
            
            # Adversarial loss for no-makeup domain
            G_loss += 0.1 * get_G_loss(makeup_remover.D, fake_nomakeup, gan_type=self.model.gan_type)
            
            # Adversarial loss for style domain (@XXX: same as style's D_loss)
            G_loss += 0.1 * get_G_loss(makeup_style_D, fake_style_pair, gan_type="gan")
            
            # Identity loss
            G_loss += F.l1_loss(real_nomakeup, makeup_remover.G(fake_makeup))
            
            # Style loss (i.e. style is preserved in fake_makeup and well-removed in fake_nomakeup)
            G_loss += F.l1_loss(real_makeup, makeup_applier.G(fake_nomakeup, fake_makeup))
            
            # Extra sparsity-inducing regularization for makeup mask
            G_loss += 0.1 * F.l1_loss(real_nomakeup, fake_makeup)

            # Calculate gradients
            G_loss.backward()

            # Make a step of minimizing G's loss
            self.optims_step("G")

            ### End of G step ###

            self._data["G_loss"].append(G_loss)

        else:
            self._data["G_loss"].append(self._data["G_loss"][-1])


    def morph_makeup(self, real_makeup, real_nomakeup, real_makeup_lm, real_nomakeup_lm):

        tensor2D_to_points = lambda t: [(p[0].item(), p[1].item()) for p in t]
        torch_to_numpy = lambda t: t.permute(1, 2, 0).numpy()
        numpy_to_torch = lambda t: torch.from_numpy(t).permute(2, 0, 1)

        batch_size = real_makeup.size()[0]
        mask = torch.ones([batch_size, 1, 1, 1]).to(real_makeup)
        morphed_batch = []

        for i in range(batch_size):
            # Zero mask for no landmarks
            if real_makeup_lm[i].sum() == 0 or real_nomakeup_lm[i].sum() == 0:
                morphed_batch.append(torch.zeros_like(real_nomakeup[i]))
                mask[i] = 0
            else:
                morphed = face_morph(torch_to_numpy(real_makeup[i]),
                                     torch_to_numpy(real_nomakeup[i]),
                                     tensor2D_to_points(real_makeup_lm[i]),
                                     tensor2D_to_points(real_nomakeup_lm[i]))
                morphed_batch.append(numpy_to_torch(morphed))
        
        return mask, torch.stack(morphed_batch).to(real_makeup)


    def checkpoint(self, epoch, num_epochs, batch, num_batches):
        # Report training stats
        if batch % self.stats_interval == 0:
            self.report_training_stats(batch, num_batches, epoch, num_epochs)


    def report_training_stats(self, epoch, num_epochs, batch, num_batches, precision=3):
        """
        Reports/prints the training stats to the console.

        Args:
            epoch: Current epoch.
            num_epochs: Max number of epochs.
            batch: Index of the current batch.
            num_batches: Max number of batches.
            precision: Precision of the float numbers reported.
        """

        report = \
            "[{epoch}/{num_epochs}][{batch}/{num_batches}]\t" \
            "Loss of D = {D_loss:.{p}f}\t" \
            "Loss of G = {G_loss:.{p}f}\t" \
            "D(x) = {D_of_x:.{p}f}\t" \
            "D(G(z)) = {D_of_G_z1:.{p}f} / {D_of_G_z2:.{p}f}"

        stats = {
            "epoch": epoch,
            "num_epochs": num_epochs,
            "batch": batch,
            "num_batches": num_batches,
            "D_loss": self._data["D_loss"][-1],
            "G_loss": self._data["G_loss"][-1],
            "p": precision,
        }

        print(report.format(**stats))


