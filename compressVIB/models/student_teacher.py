from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.metrics import softmax_accuracy
from helpers.distributions import kl_gaussian_q_N_0_1, WS22_gaussian


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

        kl = kl_gaussian_q_N_0_1(params_student['mu'], params_student['std'])
        beta = self.config['beta']

        MCsamples = len(logits_student)
        pred_list = []
        for MC_index in range(MCsamples):
            pred = F.softmax(logits_student[MC_index], dim=1)
            pred_list.append(pred)
        student_pred = sum(pred_list) / MCsamples
        ce = F.nll_loss(torch.log(student_pred), y, reduction='none')

        loss = ce + beta * kl
        acc = softmax_accuracy(student_pred, y)

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

            dzkl = WS22_gaussian(params_teacher['mu'],
                                 params_teacher['std'],
                                 params_student['mu'],
                                 params_student['std'],
                                 layer_reduction='sum')

            doutkl_list = []
            for MC_index in range(MCsamples):
                doutkl = F.kl_div(F.log_softmax(logits_student[MC_index] / T,
                                                dim=1),
                                  F.softmax(logits_teacher[MC_index] / T,
                                            dim=1),
                                  reduction='none')
                doutkl = torch.sum(doutkl, dim=1)
                doutkl_list.append(doutkl)
            doutkl = sum(doutkl_list) / MCsamples

            loss = (1. - alpha) * (ce + beta * kl) + alpha * (
                doutkl * T * T + dzkl * self.config['distill_z_kl_lambda'])

            result['loss'] = loss
            result['loss_mean'] = torch.mean(loss)
            result['dzkl_mean'] = torch.mean(dzkl)
            result['doutkl_mean'] = torch.mean(doutkl)

        return result

    def forward(self, x, MCsamples):

        if self.teacher is not None:
            logits_teacher, params_teacher = self.teacher(x,
                                                          MCsamples=MCsamples)

            logits_student, params_student = self.student(
                x, noise_list=params_teacher['noise'], MCsamples=MCsamples)

        else:

            logits_teacher, params_teacher = None, None
            logits_student, params_student = self.student(x,
                                                          MCsamples=MCsamples)

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
