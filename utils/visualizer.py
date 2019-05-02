import os

import torch
from torchvision import utils


class Visualizer():
    def __init__(self,
                 netG,
                 device,
                 out,
                 num_samples=10,
                 num_columns=None,
                 batch_size=100,
                 range=(-1, 1)):
        self.netG = netG
        self.device = device
        self.out = out
        self.num_samples = num_samples
        if num_columns is None:
            self.num_columns = self.netG.num_classes
        else:
            self.num_columns = num_columns
        self.batch_size = batch_size
        self.range = range

        z_base = netG.sample_z(num_samples).to(device)
        z = z_base.clone().unsqueeze(1).repeat(1, self.num_columns, 1)
        self.fixed_z = z.view(-1, netG.latent_dim)

    def visualize(self, iteration):
        netG = self.netG
        netG.eval()

        with torch.no_grad():
            y = torch.arange(self.num_columns).repeat(self.num_samples).to(
                self.device)
            if y.size(0) < self.batch_size:
                x = netG(self.fixed_z, y)
            else:
                xs = []
                for i in range(0, y.size(0), self.batch_size):
                    x = netG(self.fixed_z[i:i + self.batch_size],
                             y[i:i + self.batch_size])
                    xs.append(x)
                x = torch.cat(xs, dim=0)
            utils.save_image(x.detach(),
                             os.path.join(self.out,
                                          'samples_iter_%d.png' % iteration),
                             self.num_columns,
                             0,
                             normalize=True,
                             range=self.range)
