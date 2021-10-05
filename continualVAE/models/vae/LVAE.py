from __future__ import print_function
from numpy import random
import torch.nn as nn
import torch
import pprint
import torch.nn.functional as F

from models.reparameterizers.isotropic_gaussian import IsotropicGaussian
from helpers.layers import View
from helpers.distributions import nll, nll_activation, kl_gaussian, kl_gaussian_q_N_0_1, kl_out, nll_gaussian_mu_std, nll_gaussian_N_0_1
from helpers.utils import float_type

EPS_STD = 1e-6


class LVAE(nn.Module):
    def __init__(self, input_shape, encoder_arch, decoder_arch, **kwargs):
        super(LVAE, self).__init__()
        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        self.input_shape = input_shape
        self.input_chans = input_shape[0]
        self.out_chans = self.input_chans * 2 if self.config[
            'nll_type'] == 'gaussian' else self.input_chans

        # build
        arch_dict = {'1': 4, '2': 8, '3': 16, '4': 32, '5': 64, '6': 128}

        self.build_encoder(arch_dict[str(encoder_arch)])
        self.build_decoder(arch_dict[str(decoder_arch)])
        if self.config['cuda']:
            self.cuda()

    def build_encoder(self, hidden_dim):
        self.u1 = ConvRMLP(self.input_chans, 32, hidden_dim, 2, 2)
        self.u2 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        self.u3 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        self.u4 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        if self.input_shape[1] == 32:
            self.u5 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        elif self.input_shape[1] == 64:
            self.u5 = ConvRMLP_multiDS(hidden_dim,
                                       32,
                                       hidden_dim,
                                       2,
                                       2,
                                       num_downsample=2)

    def build_decoder(self, hidden_dim):
        if self.input_shape[1] == 32:
            self.d4 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        elif self.input_shape[1] == 64:
            self.d4 = ConvTRMLP_multiDS(16,
                                        32,
                                        hidden_dim,
                                        2,
                                        2,
                                        num_upsample=2)

        self.d3 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d2 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d1 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d0 = ConvTRMLP(16, self.out_chans, hidden_dim, 2, 2)

        self.d5 = nn.Linear(2, 32, bias=False)

    def iso_gauss_param(self, logits):
        feature_size = logits.size(1)
        assert feature_size % 2 == 0

        mu = logits[:, 0:int(feature_size / 2)]
        logvar = logits[:, int(feature_size / 2):]
        # if self.config['std_exp_activation']:
        #     std = logvar.mul(0.5).exp_() + EPS_STD
        # else:
        std = F.softplus(logvar) / 0.6931 + EPS_STD
        return mu, std

    def iso_gauss_sample(self, mu, std, noise=None):
        if noise is None:
            noise = float_type(self.config['cuda'])(std.size()).normal_()
        assert noise.shape == std.shape
        z = std.mul(noise).add_(mu)

        return z, noise

    def prec_weighted_com(self, mu1, std1, mu2, std2):
        prec1 = std1**(-2)
        prec2 = std2**(-2)
        std = (prec1 + prec2)**(-0.5)
        mu = (mu1 * prec1 + mu2 * prec2) / (prec1 + prec2)
        return mu, std

    def forward(self, x, cond_class):

        # upward
        # print(cond_class.shape, cond_class)
        h, l = self.u1(x, cond_class)
        um1, uv1 = self.iso_gauss_param(l)
        h, l = self.u2(h, cond_class)
        um2, uv2 = self.iso_gauss_param(l)
        h, l = self.u3(h, cond_class)
        um3, uv3 = self.iso_gauss_param(l)
        h, l = self.u4(h, cond_class)
        um4, uv4 = self.iso_gauss_param(l)
        h, l = self.u5(h, cond_class)
        um5, uv5 = self.iso_gauss_param(l)

        # downward
        l = self.d5(cond_class)
        l = l[:, :, None, None]
        pm5, pv5 = self.iso_gauss_param(l)
        qm5, qv5 = self.prec_weighted_com(um5, uv5, pm5, pv5)

        z5, n5 = self.iso_gauss_sample(qm5, qv5, None)
        _, l = self.d4(z5, cond_class)
        pm4, pv4 = self.iso_gauss_param(l)
        qm4, qv4 = self.prec_weighted_com(um4, uv4, pm4, pv4)

        z4, n4 = self.iso_gauss_sample(qm4, qv4, None)
        _, l = self.d3(z4, cond_class)
        pm3, pv3 = self.iso_gauss_param(l)
        qm3, qv3 = self.prec_weighted_com(um3, uv3, pm3, pv3)

        z3, n3 = self.iso_gauss_sample(qm3, qv3, None)
        _, l = self.d2(z3, cond_class)
        pm2, pv2 = self.iso_gauss_param(l)
        qm2, qv2 = self.prec_weighted_com(um2, uv2, pm2, pv2)

        z2, n2 = self.iso_gauss_sample(qm2, qv2, None)
        _, l = self.d1(z2, cond_class)
        pm1, pv1 = self.iso_gauss_param(l)
        qm1, qv1 = self.prec_weighted_com(um1, uv1, pm1, pv1)

        z1, n1 = self.iso_gauss_sample(qm1, qv1, None)
        _, logits = self.d0(z1, cond_class)

        # logits = logits.view(batch_size, *self.input_shape)

        params = {
            'q': [(qm1, qv1), (qm2, qv2), (qm3, qv3), (qm4, qv4), (qm5, qv5)],
            'p': [(pm1, pv1), (pm2, pv2), (pm3, pv3), (pm4, pv4), (pm5, pv5)],
            'z': [z1, z2, z3, z4, z5],
            'noise': [n1, n2, n3, n4, n5]
        }

        # print(qm4.shape, pm4.shape, qm5.shape)

        return logits, params

    def nelbo(self, x, logits, params, beta):
        # print(x.max(), x.min(), logits.max(), logits.min(), flush=True)
        elbo_nll = nll(x, logits, self.config['nll_type'])
        kl5 = kl_gaussian(*params['q'][4], *params['p'][4])
        kl4 = kl_gaussian(*params['q'][3], *params['p'][3])
        kl3 = kl_gaussian(*params['q'][2], *params['p'][2])
        kl2 = kl_gaussian(*params['q'][1], *params['p'][1])
        kl1 = kl_gaussian(*params['q'][0], *params['p'][0])
        kl = kl1 + kl2 + kl3 + kl4 + kl5
        nelbo = elbo_nll + beta * kl
        return nelbo, elbo_nll, kl1, kl2, kl3, kl4, kl5

    def generate_cond_class(self, size, class_begin, class_end):

        num_class = class_end - class_begin
        real_size = size // num_class * num_class
        cond_class = float_type(self.config['cuda'])(real_size, 2).zero_()

        # r_list = torch.randint(0, num_class, size)
        r_list = torch.arange(0, real_size, dtype=torch.long
                              ) * num_class // real_size + class_begin

        for i in range(len(r_list)):
            cond_class[i, r_list[i]] = 1
        return cond_class

    def generate_synthetic_samples(self,
                                   size,
                                   class_begin,
                                   class_end,
                                   z_list=None,
                                   noise_list=None):
        if z_list is not None:
            z1, z2, z3, z4, z5 = z_list
        if noise_list is not None:
            n1, n2, n3, n4, n5 = noise_list
        else:
            n1 = n2 = n3 = n4 = n5 = None

        cond_class = self.generate_cond_class(size, class_begin, class_end)

        l = self.d5(cond_class)
        l = l[:, :, None, None]
        pm5, pv5 = self.iso_gauss_param(l)

        if z_list is None:
            z5, n5 = self.iso_gauss_sample(pm5, pv5, n5)
        _, l = self.d4(z5, cond_class)
        pm4, pv4 = self.iso_gauss_param(l)

        if z_list is None:
            z4, n4 = self.iso_gauss_sample(pm4, pv4, n4)
        _, l = self.d3(z4, cond_class)
        pm3, pv3 = self.iso_gauss_param(l)

        if z_list is None:
            z3, n3 = self.iso_gauss_sample(pm3, pv3, n3)
        _, l = self.d2(z3, cond_class)
        pm2, pv2 = self.iso_gauss_param(l)

        if z_list is None:
            z2, n2 = self.iso_gauss_sample(pm2, pv2, n2)
        _, l = self.d1(z2, cond_class)
        pm1, pv1 = self.iso_gauss_param(l)

        if z_list is None:
            z1, n1 = self.iso_gauss_sample(pm1, pv1, n1)
        _, logits = self.d0(z1, cond_class)

        # logits = l.view(batch_size, *self.input_shape)

        params = {
            'p': [(pm1, pv1), (pm2, pv2), (pm3, pv3), (pm4, pv4), (pm5, pv5)],
            'z': [z1, z2, z3, z4, z5],
            'noise': [n1, n2, n3, n4, n5]
        }

        return logits, nll_activation(logits, self.config['nll_type']), params

    def reconstruct(self, x):
        z_logits = self.encoder(x)

        z, _ = self.reparameterizer(z_logits, 0)

        logits = self.decoder(z)

        return nll_activation(logits, self.config['nll_type'], binarize=True)

    def importance_sampling_prob(self, x, num_samples):
        # upward once
        h, l = self.u1(x)
        um1, uv1 = self.iso_gauss_param(l)
        h, l = self.u2(h)
        um2, uv2 = self.iso_gauss_param(l)
        h, l = self.u3(h)
        um3, uv3 = self.iso_gauss_param(l)
        h, l = self.u4(h)
        um4, uv4 = self.iso_gauss_param(l)
        h, l = self.u5(h)
        um5, uv5 = self.iso_gauss_param(l)

        # downward many times
        prob_list = []
        for sample_index in range(num_samples):
            # downward
            qm5, qv5 = um5, uv5

            z5, _ = self.iso_gauss_sample(qm5, qv5)
            _, l = self.d4(z5)
            pm4, pv4 = self.iso_gauss_param(l)
            qm4, qv4 = self.prec_weighted_com(um4, uv4, pm4, pv4)

            z4, _ = self.iso_gauss_sample(qm4, qv4)
            _, l = self.d3(z4)
            pm3, pv3 = self.iso_gauss_param(l)
            qm3, qv3 = self.prec_weighted_com(um3, uv3, pm3, pv3)

            z3, _ = self.iso_gauss_sample(qm3, qv3)
            _, l = self.d2(z3)
            pm2, pv2 = self.iso_gauss_param(l)
            qm2, qv2 = self.prec_weighted_com(um2, uv2, pm2, pv2)

            z2, _ = self.iso_gauss_sample(qm2, qv2)
            _, l = self.d1(z2)
            pm1, pv1 = self.iso_gauss_param(l)
            qm1, qv1 = self.prec_weighted_com(um1, uv1, pm1, pv1)

            z1, _ = self.iso_gauss_sample(qm1, qv1)
            _, logits = self.d0(z1)

            nlpxz = nll(x, logits, self.config['nll_type'])
            nlqzx = nll_gaussian_mu_std(z1, qm1, qv1) + nll_gaussian_mu_std(
                z2, qm2,
                qv2) + nll_gaussian_mu_std(z3, qm3, qv3) + nll_gaussian_mu_std(
                    z4, qm4, qv4) + nll_gaussian_mu_std(z5, qm5, qv5)

            nlpz = nll_gaussian_mu_std(z1, pm1, pv1) + nll_gaussian_mu_std(
                z2, pm2,
                pv2) + nll_gaussian_mu_std(z3, pm3, pv3) + nll_gaussian_mu_std(
                    z4, pm4, pv4) + nll_gaussian_N_0_1(z5)

            prob_list.append(torch.exp(-nlpz - nlpxz + nlqzx))

        prob = sum(prob_list)
        if not torch.all(prob > 0):
            print('importance sampling -inf occured, too low is the prob')
            return 999
        return -torch.log(prob)

    def train(self, mode: bool = True):
        res = super().train(mode=mode)
        for name, m in self.named_modules():
            if 'bn' in name:
                m.train(True)
        return res


class ConvRMLP(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=8,
                 hidden_dim=32,
                 stride=2,
                 num_block=1):
        super(ConvRMLP, self).__init__()
        blocks = []

        # downsample are treated at beggining
        blocks.append(ConvBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvRMLP_multiDS(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=8,
                 hidden_dim=32,
                 stride=2,
                 num_block=2,
                 num_downsample=2):
        super(ConvRMLP_multiDS, self).__init__()
        blocks = []

        # downsample are treated at beggining
        blocks.append(ConvBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))
        for index_ds in range(1, num_downsample):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim, stride))
            for _ in range(1, num_block):
                blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvTRMLP(nn.Module):
    def __init__(self,
                 in_dim=4,
                 out_dim=1,
                 hidden_dim=32,
                 stride=2,
                 num_block=1):
        super(ConvTRMLP, self).__init__()
        blocks = []

        # upsample are treated at beggining
        blocks.append(ConvTBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvTRMLP_multiDS(nn.Module):
    def __init__(self,
                 in_dim=4,
                 out_dim=1,
                 hidden_dim=32,
                 stride=2,
                 num_block=2,
                 num_upsample=2):
        super(ConvTRMLP_multiDS, self).__init__()
        blocks = []

        # upsample are treated at beggining
        blocks.append(ConvTBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))
        for index_ds in range(1, num_upsample):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim, stride))
            for _ in range(1, num_block):
                blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvTBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvTBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.unpool = None
        if stride != 1:
            self.unpool = nn.UpsamplingNearest2d(scale_factor=stride)
        self.upsample = None
        if stride != 1 or inplanes != planes:
            self.upsample = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.unpool is not None:
            x = self.unpool(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvBasicBlock_Cond(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvBasicBlock_Cond, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        inplanes = inplanes + 2
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x_cond_class):
        x, cond_class = x_cond_class
        cond_class_ext = cond_class[:, :, None, None].repeat(
            (1, 1, x.shape[2], x.shape[3]))
        x = torch.cat([x, cond_class_ext], dim=1)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvTBasicBlock_Cond(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvTBasicBlock_Cond, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        inplanes = inplanes + 2
        self.unpool = None
        if stride != 1:
            self.unpool = nn.UpsamplingNearest2d(scale_factor=stride)
        self.upsample = None
        if stride != 1 or inplanes != planes:
            self.upsample = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x_cond_class):
        x, cond_class = x_cond_class
        cond_class_ext = cond_class[:, :, None, None].repeat(
            (1, 1, x.shape[2], x.shape[3]))
        x = torch.cat([x, cond_class_ext], dim=1)
        if self.unpool is not None:
            x = self.unpool(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
