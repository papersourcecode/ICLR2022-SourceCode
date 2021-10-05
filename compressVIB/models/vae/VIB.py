from __future__ import print_function
import torch.nn as nn
import torch
import pprint
import torch.nn.functional as F

from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from helpers.layers import View
from helpers.distributions import nll, nll_activation, kl_gaussian, kl_gaussian_q_N_0_1, kl_out, nll_gaussian_mu_std, nll_gaussian_N_0_1
from helpers.utils import float_type
from helpers.resnet_cifar import resnet32, resnet20
from helpers.resnet_imagenet import resnet18

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
        if arch == 3:
            self.net = nn.Sequential(View([-1, input_dim]),
                                     nn.Linear(input_dim, 512), nn.ReLU(True),
                                     nn.Linear(512, 512), nn.ReLU(),
                                     nn.Linear(512, 512))

        if arch == 4:
            self.net = nn.Sequential(View([-1, input_dim]),
                                     nn.Linear(input_dim, 128), nn.ReLU(True),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 512))

        self.linear2 = nn.Linear(256, output_classes)
        self.dropout = nn.Dropout(0.5)

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

    def forward(self,
                x,
                z_list=None,
                noise_list=None,
                MCsamples=1,
                mode='VIB'):

        feature = self.net(x)
        if mode == 'dropout':
            feature = self.dropout(feature)
        mu, std = self.iso_gauss_param(feature)

        if mode == 'VIB':
            # MC sample many times
            z_list_out = []
            noise_list_out = []
            logits_list = []
            for MC_index in range(MCsamples):

                # handling shared noise or z
                if z_list is None:
                    if noise_list is None:
                        noise = None
                    else:
                        noise = noise_list[MC_index]
                    z, noise = self.iso_gauss_sample(mu, std, noise)
                else:
                    z, noise = z_list[MC_index], None
                logits = self.linear2(z)
                z_list_out.append(z)
                noise_list_out.append(noise)
                logits_list.append(logits)

            params = {
                'z': z_list_out,
                'noise': noise_list_out,
                'mu': mu,
                'std': std
            }

            return logits_list, params
        else:
            logits = self.linear2(mu)

            return logits, None
