from loss import utility
from model import discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k

    def forward(self, fake, real, discriminator):
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            d_fake = fake.detach()
            d_real = real
            if self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = Variable(
                        fake_detach.data.new(fake.size(0), 1, 1, 1).uniform_(),
                        requires_grad=False
                    )
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            if self.gan_type == 'WGAN':
                for p in discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        d_fake_for_g = discriminator(fake)
        if self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()

        return loss_g, loss_d #####

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss

