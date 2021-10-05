from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

from helpers.utils import float_type, zeros_like, ones_like


class IsotropicGaussian(nn.Module):
    ''' isotropic gaussian reparameterization '''
    def __init__(self, input_size, config):
        super(IsotropicGaussian, self).__init__()
        self.config = config
        self.input_size = input_size
        assert input_size % 2 == 0
        self.output_size = input_size // 2

    def prior(self, batch_size, **kwargs):
        scale_var = 1.0 if 'scale_var' not in kwargs else kwargs['scale_var']
        return Variable(
            float_type(self.config['cuda'])(
                batch_size, self.output_size).normal_(mean=0, std=scale_var))

    def forward(self, logits, noise=None):

        feature_size = logits.size(-1)
        assert feature_size % 2 == 0 and feature_size // 2 == self.output_size

        mu = logits[:, 0:int(feature_size / 2)]
        logvar = logits[:, int(feature_size / 2):]
        std = logvar.mul(0.5).exp_()

        if noise is None:
            noise = float_type(self.config['cuda'])(std.size()).normal_()

        z = std.mul(noise).add_(mu)
        return z, {'noise': noise, 'mu': mu, 'logvar': logvar}
