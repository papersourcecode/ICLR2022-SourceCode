from __future__ import print_function
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from copy import deepcopy

from helpers.metrics import softmax_accuracy
from helpers.distributions import kl_gaussian_q_N_0_1, nll, nll_activation, kl_gaussian, kl_out, WS22_gaussian, kl_gaussian_q_p, prob_ratio_gaussian
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

        # grab the meta config and print for
        self.config = kwargs['kwargs']

    def loss_function(self, y, output_map):

        logits_student = output_map['student']['logits']
        params_student = output_map['student']['params']
        logits_teacher = output_map['teacher']['logits']
        params_teacher = output_map['teacher']['params']

        if self.config['mode'] in ('VIB', 'our'):

            # VIB kl loss
            kl = kl_gaussian_q_N_0_1(params_student['mu'],
                                     params_student['std'])
            beta = self.config['beta']

            # VIB student pred and ce loss
            MCsamples = len(logits_student)
            pred_list = []
            for MC_index in range(MCsamples):
                pred = F.softmax(logits_student[MC_index], dim=1)
                pred_list.append(pred)
            student_pred = sum(pred_list) / MCsamples
            ce = F.nll_loss(torch.log(student_pred), y, reduction='none')

            loss = ce + beta * kl
            acc = softmax_accuracy(student_pred, y)

        elif self.config['mode'] in ('vanilla', 'dropout'):
            # student pred and ce loss
            student_pred = F.softmax(logits_student, dim=1)
            ce = F.cross_entropy(logits_student, y, reduction='none')
            loss = ce

            kl = torch.zeros_like(ce)
            acc = softmax_accuracy(logits_student, y)

        else:
            raise NotImplementedError('bug')

        result = {
            'loss': loss,
            'loss_mean': torch.mean(loss),
            'ce_mean': torch.mean(ce),
            'kl_mean': torch.mean(kl),
            'acc_mean': acc,
        }

        if self.teacher is not None:
            alpha = self.config['distill_out_kl_lambda']
            T = self.config['temperature']

            if self.config['mode'] == 'our':

                if self.config['distill_KL']:
                    dissimilarity = kl_gaussian
                else:
                    dissimilarity = WS22_gaussian

                dzkl = dissimilarity(params_teacher['mu'],
                                     params_teacher['std'],
                                     params_student['mu'],
                                     params_student['std'],
                                     layer_reduction='sum')

                doutkl_list = []
                for MC_index in range(MCsamples):
                    doutkl = F.kl_div(F.log_softmax(logits_student[MC_index] /
                                                    T,
                                                    dim=1),
                                      F.softmax(logits_teacher[MC_index] / T,
                                                dim=1),
                                      reduction='none')
                    doutkl = torch.sum(doutkl, dim=1)
                    doutkl_list.append(doutkl)
                doutkl = sum(doutkl_list) / MCsamples

                loss = (1. - alpha) * (ce + beta * kl) + alpha * (
                    doutkl * T * T + dzkl * self.config['distill_z_kl_lambda'])

            elif self.config['mode'] in ('VIB', 'vanilla', 'dropout'):
                # teacher pred
                MCsamples = len(logits_teacher)
                pred_list = []
                for MC_index in range(MCsamples):
                    pred = F.softmax(logits_teacher[MC_index], dim=1)
                    pred_list.append(pred)
                teacher_pred = sum(pred_list) / MCsamples

                # student pred has already calced
                # out kl loss
                soft_t_pred = teacher_pred**(1 / T)
                soft_t_pred = soft_t_pred / torch.sum(soft_t_pred, dim=1)[:,
                                                                          None]
                soft_s_pred = student_pred**(1 / T)
                soft_s_pred = soft_s_pred / torch.sum(soft_s_pred, dim=1)[:,
                                                                          None]

                doutkl = F.kl_div(torch.log(soft_t_pred),
                                  soft_s_pred,
                                  reduction='none')
                doutkl = torch.sum(doutkl, dim=1)

                loss = doutkl * T * T * alpha + (1. - alpha) * ce
                dzkl = torch.zeros_like(doutkl)

            result['loss'] = loss
            result['loss_mean'] = torch.mean(loss)
            result['dzkl_mean'] = torch.mean(dzkl)
            result['doutkl_mean'] = torch.mean(doutkl)

        return result

    def forward(self, x, MCsamples):

        if self.teacher is not None:
            # distillation
            # only use VIB as teacher
            logits_teacher, params_teacher = self.teacher(x,
                                                          MCsamples=MCsamples,
                                                          mode='VIB')

            if self.config['mode'] == 'our':

                if self.config['distill_share_z']:
                    logits_student, params_student = self.student(
                        x,
                        z_list=params_teacher['z'],
                        MCsamples=MCsamples,
                        mode='VIB')
                else:
                    logits_student, params_student = self.student(
                        x,
                        noise_list=params_teacher['noise'],
                        MCsamples=MCsamples,
                        mode='VIB')
            elif self.config['mode'] in ('VIB', 'vanilla', 'dropout'):
                logits_student, params_student = self.student(
                    x, MCsamples=MCsamples, mode=self.config['mode'])
            else:
                raise NotImplementedError('bug')
        else:
            # scratch
            if self.config['mode'] in ('VIB', 'vanilla', 'dropout'):
                logits_teacher, params_teacher = None, None
                logits_student, params_student = self.student(
                    x, MCsamples=MCsamples, mode=self.config['mode'])
            else:
                raise NotImplementedError('bug')

        ret_map = {
            'student': {
                'params': params_student,
                'logits': logits_student
            },
            'teacher': {
                'params': params_teacher,
                'logits': logits_teacher
            }
        }

        return ret_map
