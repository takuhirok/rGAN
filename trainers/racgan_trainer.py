import torch
import torch.nn.functional as F

from utils import util


class rACGANTrainer():
    def __init__(self, *args, **kwargs):
        self.iterator = kwargs.pop('iterator')
        self.netG, self.netD = kwargs.pop('models')
        self.optimizerG, self.optimizerD = kwargs.pop('optimizers')
        self.gan_loss = kwargs.pop('gan_loss', 'gan')
        self.lr_schedulers = kwargs.pop('lr_schedulers')
        self.batch_size = kwargs.pop('batch_size')
        self.g_bs_multiple = kwargs.pop('g_bs_multiple', 1)
        self.num_critic = kwargs.pop('num_critic', 1)
        (self.lambda_gp, self.lambda_ct, self.lambda_cls_g,
         self.lambda_cls_d) = kwargs.pop('lambdas', (0., 0., 1., 1.))
        self.factor_m = kwargs.pop('factor_m', 0.)
        self.noise2clean = kwargs.pop('noise2clean', lambda x: x)
        self.T = kwargs.pop('T', None)
        self.device = kwargs.pop('device')

        self.loss = {}
        self.iteration = 0

    def gradient_penalty(self, x_real, x_fake, netD, index=0):
        device = x_real.device
        with torch.enable_grad():
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
            alpha = alpha.expand_as(x_real)
            x_hat = alpha * x_real.data + (1 - alpha) * x_fake.data
            x_hat.requires_grad = True
            output = netD(x_hat)[index]
            grad_output = torch.ones(output.size()).to(device)
            grad = torch.autograd.grad(outputs=output,
                                       inputs=x_hat,
                                       grad_outputs=grad_output,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            grad = grad.view(grad.size(0), -1)
            loss_gp = ((grad.norm(p=2, dim=1) - 1)**2).mean()
        return loss_gp

    def update(self):
        netG, netD = self.netG, self.netD
        netG.train()
        netD.train()

        optimizerG, optimizerD = self.optimizerG, self.optimizerD

        if self.iteration > 0:
            # Train G
            g_batch_size = self.g_bs_multiple * self.batch_size
            z = netG.sample_z(g_batch_size).to(self.device)
            y_noisy = torch.cat([self.label_noisy] * self.g_bs_multiple,
                                dim=0).to(self.device)
            y_clean = self.noise2clean(y_noisy)
            x_fake = netG(z, y_clean)
            out_dis, out_cls, _ = netD(x_fake)
            if self.gan_loss == 'wgan':
                g_loss_fake = -out_dis.mean()
            elif self.gan_loss == 'hinge':
                g_loss_fake = -out_dis.mean()
            else:
                g_loss_fake = F.binary_cross_entropy_with_logits(
                    out_dis,
                    torch.ones(g_batch_size).to(self.device))

            g_loss_cls = self.lambda_cls_g * F.cross_entropy(out_cls, y_clean)

            g_loss = 0
            g_loss = g_loss + g_loss_fake
            g_loss = g_loss + g_loss_cls
            netG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            self.loss['G/loss_fake'] = g_loss_fake.item()
            self.loss['G/loss_cls'] = g_loss_cls.item()

        # Train D
        for i in range(self.num_critic):
            image_real, label_noisy = next(self.iterator)
            x_real = image_real.to(self.device)
            y_noisy = label_noisy.to(self.device)
            out_dis, out_cls, out_feat = netD(x_real)
            if self.gan_loss == 'wgan':
                d_loss_real = -out_dis.mean()
            elif self.gan_loss == 'hinge':
                d_loss_real = (F.relu(1. - out_dis)).mean()
            else:
                d_loss_real = F.binary_cross_entropy_with_logits(
                    out_dis,
                    torch.ones(self.batch_size).to(self.device))

            if self.T is None:
                d_loss_cls = self.lambda_cls_d * F.cross_entropy(
                    out_cls, y_noisy)
            else:
                eps = 1e-8
                p = F.softmax(out_cls, dim=1)
                d_loss_cls = -self.lambda_cls_d * (torch.sum(
                    util.one_hot(y_noisy, netD.num_classes) *
                    torch.log(p.mm(self.T) + eps)) / y_noisy.size(0))

            if self.lambda_ct > 0:
                out_dis_, _, out_feat_ = netD(x_real)
                d_loss_ct = self.lambda_ct * (out_dis - out_dis_)**2
                d_loss_ct = d_loss_ct + (self.lambda_ct * 0.1 *
                                         ((out_feat - out_feat_)**2).mean(1))
                d_loss_ct = torch.max(d_loss_ct - self.factor_m,
                                      0.0 * (d_loss_ct - self.factor_m))
                d_loss_ct = d_loss_ct.mean()

            z = netG.sample_z(self.batch_size).to(self.device)
            y_clean = self.noise2clean(y_noisy)
            x_fake = netG(z, y_clean)
            out_dis, _, _ = netD(x_fake.detach())
            if self.gan_loss == 'wgan':
                d_loss_fake = out_dis.mean()
            elif self.gan_loss == 'hinge':
                d_loss_fake = (F.relu(1. + out_dis)).mean()
            else:
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    out_dis,
                    torch.zeros(self.batch_size).to(self.device))

            if self.lambda_gp > 0:
                d_loss_gp = self.lambda_gp * self.gradient_penalty(
                    x_real, x_fake, netD)

            d_loss = 0
            d_loss = d_loss + d_loss_real + d_loss_fake
            d_loss = d_loss + d_loss_cls
            if self.lambda_gp > 0:
                d_loss = d_loss + d_loss_gp
            if self.lambda_ct > 0:
                d_loss = d_loss + d_loss_ct
            netD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            if self.gan_loss == 'wgan' or self.gan_loss == 'hinge':
                self.loss['D/loss_adv'] = (d_loss_real.item() +
                                           d_loss_fake.item())
            else:
                self.loss['D/loss_real'] = d_loss_real.item()
                self.loss['D/loss_fake'] = d_loss_fake.item()
            self.loss['D/loss_cls'] = d_loss_cls.item()
            if self.lambda_gp > 0:
                self.loss['D/loss_gp'] = d_loss_gp.item()
            if self.lambda_ct > 0:
                self.loss['D/loss_ct'] = d_loss_ct.item()
            self.loss['D/loss'] = d_loss.item()

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        self.label_noisy = label_noisy
        self.iteration += 1

    def get_current_loss(self):
        return self.loss
