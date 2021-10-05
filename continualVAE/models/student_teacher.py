from __future__ import print_function
import os
from warnings import resetwarnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from copy import deepcopy

from helpers.distributions import nll, nll_activation, kl_gaussian, kl_out, WS22_gaussian, kl_gaussian_q_p, prob_ratio_gaussian
from helpers.utils import expand_dims, long_type, squeeze_expand_dim, \
    ones_like, float_type, pad, inv_perm, one_hot_np, \
    zero_pad_smaller_cat, check_or_create_dir


def detach_from_graph(param_map):
    for _, v in param_map.items():
        if isinstance(v, dict):
            detach_from_graph(v)
        else:
            v = v.detach_()


class StudentTeacher(nn.Module):
    def __init__(self, teacher_model, student_model, **kwargs):
        ''' Helper to keep the student-teacher architecture '''
        super(StudentTeacher, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.continual_step = 0

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def loss_function(self, output_map, beta, alpha):
        result = {}

        x = output_map['x']
        x_reconstr_logits_student = output_map['student']['x_reconstr_logits']
        params_student = output_map['student']['params']

        elbo, elbo_nll, kl1, kl2, kl3, kl4, kl5 = self.student.nelbo(
            x, x_reconstr_logits_student, params_student, beta)

        loss = elbo
        if 'replay' in output_map:
            x = output_map['gen_teacher']
            x_reconstr_logits_student = output_map['replay'][
                'x_reconstr_logits']
            params_student = output_map['replay']['params']

            elbo_replay, _, _, _, _, _, _ = self.student.nelbo(
                x, x_reconstr_logits_student, params_student, beta)

            if self.config['bayesian_update'] == 1:
                params_teacher = output_map['teacher_params']
                bulosses = []
                for layer_index in (0, 1, 2, 3, 4):

                    buloss = kl_gaussian(*params_student['q'][layer_index],
                                         *params_teacher['q'][layer_index],
                                         layer_reduction='sum')

                    bulosses.append(buloss)

                buloss = sum(bulosses)
                elbo_replay = elbo_replay + elbo_replay

            loss = torch.cat((elbo * self.config['ratio'], elbo_replay *
                              (1 - self.config['ratio'])),
                             dim=0)

        result.update({
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'elbo_mean': torch.mean(elbo),
            'elbo_nll_mean': torch.mean(elbo_nll),
            'kl1_mean': torch.mean(kl1),
            'kl2_mean': torch.mean(kl2),
            'kl3_mean': torch.mean(kl3),
            'kl4_mean': torch.mean(kl4),
            'kl5_mean': torch.mean(kl5)
        })

        if 'distill' in output_map:
            gen_logits_teacher = output_map['distill']['gen_logits_teacher']
            gen_logits_student = output_map['distill']['gen_logits_student']
            params_gen_teacher = output_map['distill']['params_gen_teacher']
            params_gen_student = output_map['distill']['params_gen_student']

            if self.config['distill_KL'] == 1:
                dissimilarity = kl_gaussian
            else:
                dissimilarity = WS22_gaussian
            # dqkl5 = dissimilarity(*params_student['q'][4],
            #                       *params_teacher['q'][4],
            #                       layer_reduction='mean')

            # dqkl = []
            dpkl = []
            for layer_index in (0, 1, 2, 3):
                # diss_q = dissimilarity(*params_student['q'][layer_index],
                #                        *params_teacher['q'][layer_index],
                #                        layer_reduction='mean')
                diss_p = dissimilarity(
                    *params_gen_teacher['p'][layer_index],
                    *params_gen_student['p'][layer_index],
                    layer_reduction=self.config['distill_z_reduction'])
                # importance_weight = prob_ratio_gaussian(
                #     params_student['z'][layer_index + 1],
                #     *params_teacher['q'][layer_index + 1],
                #     *params_student['q'][layer_index + 1])
                importance_weight = 1
                # dqkl.append(diss_q * importance_weight)
                dpkl.append(diss_p * importance_weight)
                # if (importance_weight > 0).any():
                #     index = importance_weight > 0
                #     print(diss_p[index], diss_q[index],
                #           importance_weight[index])

            # dqkl1, dqkl2, dqkl3, dqkl4 = dqkl
            # dpkl1, dpkl2, dpkl3, dpkl4 = dpkl

            doutkl = kl_out(gen_logits_student, gen_logits_teacher,
                            self.config['nll_type'])

            dzkl = sum(dpkl)

            distill_loss = (doutkl +
                            self.config['distill_z_kl_lambda'] * dzkl) * alpha
            loss = torch.cat(
                (distill_loss *
                 (1 - self.config['ratio']), elbo * self.config['ratio']),
                dim=0)

            result.update({
                'loss': loss,
                'loss_mean': torch.mean(loss),
                'distill_mean': torch.mean(distill_loss),
                'dpkl1_mean': torch.mean(dpkl[0]),
                'dpkl2_mean': torch.mean(dpkl[1]),
                'dpkl3_mean': torch.mean(dpkl[2]),
                'dpkl4_mean': torch.mean(dpkl[3]),
                'doutkl_mean': torch.mean(doutkl)
            })

        return result

    def forward(self, x):
        cond_class = self.student.generate_cond_class(x.shape[0],
                                                      self.continual_step,
                                                      self.continual_step + 1)
        x_reconstr_logits, params_student = self.student(x, cond_class)
        x_reconstr = nll_activation(x_reconstr_logits, self.config['nll_type'])

        ret_map = {
            'student': {
                'params': params_student,
                'x_reconstr': x_reconstr,
                'x_reconstr_logits': x_reconstr_logits
            },
            'x': x,
        }

        return ret_map

    def distill_forward(self, x):
        cond_class = self.student.generate_cond_class(x.shape[0],
                                                      self.continual_step,
                                                      self.continual_step + 1)
        x_reconstr_logits, params_student = self.student(x, cond_class)
        x_reconstr = nll_activation(x_reconstr_logits, self.config['nll_type'])

        ret_map = {
            'student': {
                'params': params_student,
                'x_reconstr': x_reconstr,
                'x_reconstr_logits': x_reconstr_logits
            },
            'x': x,
        }

        gen_size = self.config['batch_size']
        gen_logits_teacher, gen_teacher, params_gen_teacher = self.teacher.generate_synthetic_samples(
            gen_size, 0, self.continual_step)

        # distill
        # student generate samples in its eval mode
        # training = self.student.training
        # self.student.eval()
        if self.config['distill_share_z'] == 1:
            gen_logits_student, gen_student, params_gen_student = self.student.generate_synthetic_samples(
                gen_size,
                0,
                self.continual_step,
                z_list=params_gen_teacher['z'])
        else:
            gen_logits_student, gen_student, params_gen_student = self.student.generate_synthetic_samples(
                gen_size,
                0,
                self.continual_step,
                noise_list=params_gen_teacher['noise'])
        # self.student.train(training)

        ret_map.update({
            'distill': {
                'gen_logits_teacher': gen_logits_teacher,
                'gen_logits_student': gen_logits_student,
                'gen_teacher': gen_teacher,
                'gen_student': gen_student,
                'params_gen_teacher': params_gen_teacher,
                'params_gen_student': params_gen_student
            },
        })
        return ret_map

    def replay_forward(self, x):

        cond_class = self.student.generate_cond_class(x.shape[0],
                                                      self.continual_step,
                                                      self.continual_step + 1)
        x_reconstr_logits, params_student = self.student(x, cond_class)
        x_reconstr = nll_activation(x_reconstr_logits, self.config['nll_type'])

        gen_size = self.config['batch_size']
        _, gen_teacher, _ = self.teacher.generate_synthetic_samples(
            gen_size, 0, self.continual_step)

        cond_class = self.student.generate_cond_class(x.shape[0], 0,
                                                      self.continual_step)
        x_reconstr_logits_gen, params_student_gen = self.student(
            gen_teacher, cond_class)
        x_reconstr_gen = nll_activation(x_reconstr_logits_gen,
                                        self.config['nll_type'])

        _, params_teacher_gen = self.teacher(gen_teacher, cond_class)

        ret_map = {
            'student': {
                'params': params_student,
                'x_reconstr': x_reconstr,
                'x_reconstr_logits': x_reconstr_logits
            },
            'x': x,
            'replay': {
                'params': params_student_gen,
                'x_reconstr': x_reconstr_gen,
                'x_reconstr_logits': x_reconstr_logits_gen
            },
            'gen_teacher': gen_teacher,
            'teacher_params': params_teacher_gen,
        }

        return ret_map
