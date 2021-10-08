from __future__ import print_function
import torch.nn as nn
import torch
import pprint
import torch.nn.functional as F

from helpers.layers import View
from helpers.utils import float_type

EPS_STD = 1e-6


class VIB(nn.Module):
    def __init__(self, input_shape, output_classes, arch, **kwargs):
        super(VIB, self).__init__()
        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        self.input_shape = input_shape

        input_dim = input_shape[1] * input_shape[2]

        if arch == 1:
            self.net = nn.Sequential(View([-1, input_dim]),
                                     nn.Linear(input_dim, 1024), nn.ReLU(True),
                                     nn.Linear(1024, 1024), nn.ReLU(),
                                     nn.Linear(1024, 512))
        if arch == 2:
            self.net = nn.Sequential(View([-1, input_dim]),
                                     nn.Linear(input_dim, 256), nn.ReLU(True),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 512))

        self.linear2 = nn.Linear(256, output_classes)

        if self.config['cuda']:
            self.cuda()

    def iso_gauss_param(self, logits):
        feature_size = logits.size(1)
        assert feature_size % 2 == 0

        mu = logits[:, 0:int(feature_size / 2)]
        logvar = logits[:, int(feature_size / 2):]
        # if self.config['std_exp_activation']:
        #     std = logvar.mul(0.5).exp_() + EPS_STD
        # else:
        std = F.softplus(logvar - 5.0) + EPS_STD
        return mu, std

    def iso_gauss_sample(self, mu, std, noise=None):
        if noise is None:
            noise = float_type(self.config['cuda'])(std.size()).normal_()
        assert noise.shape == std.shape
        z = std.mul(noise).add_(mu)

        return z, noise

    def forward(self, x, noise_list=None, MCsamples=1):

        feature = self.net(x)
        mu, std = self.iso_gauss_param(feature)

        # MC sample many times
        noise_list_out = []
        logits_list = []
        for MC_index in range(MCsamples):

            if noise_list is None:
                noise = None
            else:
                noise = noise_list[MC_index]
            z, noise = self.iso_gauss_sample(mu, std, noise)

            logits = self.linear2(z)
            noise_list_out.append(noise)
            logits_list.append(logits)

        params = {'noise': noise_list_out, 'mu': mu, 'std': std}

        return logits_list, params
