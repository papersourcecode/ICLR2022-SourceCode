import os
import argparse
import pprint
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.vae.LVAE import HM
from models.student_teacher import StudentTeacher
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.utils import number_of_parameters
import GPUs

parser = argparse.ArgumentParser(description='VAE distillation Pytorch')

# Task parameters
parser.add_argument(
    '--uid',
    type=str,
    default="",
    help="add a custom task-specific unique id (default: None)")
parser.add_argument('--epochs',
                    type=int,
                    default=4000,
                    metavar='N',
                    help='minimum number of epochs to train (default: 10)')

parser.add_argument('--ckpt-dir',
                    type=str,
                    default='./CKPT',
                    metavar='OD',
                    help='directory which contains ckpt')

# Optimization related
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help="specify optimizer (default: SGD)")
parser.add_argument('--lr',
                    type=float,
                    default=1e-2,
                    metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size',
                    type=int,
                    default=272,
                    metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--mode',
                    type=str,
                    default='our',
                    help='scratch / local / our')
parser.add_argument('--eval', action='store_true', default=False, help='help')
parser.add_argument('--resume', type=int, default=0, help='help')
# Visdom parameters
parser.add_argument('--visdom-url',
                    type=str,
                    default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port',
                    type=int,
                    default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
parser.add_argument('--seed',
                    type=int,
                    default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu',
                    type=int,
                    default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--gpu-wait', type=float, default=1.0, help='wait until')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
args = parser.parse_args()

Y1_MEAN = 3.48778309
Y2_MEAN = 70.89705882
Y1_STD = 1.13927121
Y2_STD = 13.56996002


class OFDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('./OF.txt', 'r') as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            line = line.strip('\n').split()
            y1 = (float(line[1]) - Y1_MEAN) / Y1_STD
            y2 = (float(line[2]) - Y2_MEAN) / Y2_STD

            d = [y1, y2]
            self.data += [d]

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError

        d = self.data[idx]
        y1 = d[0:1]
        y2 = d[1:2]

        return torch.tensor(y1).cuda(), torch.tensor(y2).cuda()


def parameters_grad_to_vector(parameters) -> torch.Tensor:

    vec = []
    for param in parameters:
        if param.grad is not None:
            vec.append(param.grad.view(-1))
    return torch.cat(vec)


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](model.parameters(),
                                                     lr=args.lr,
                                                     weight_decay=1e-4)


def register_plots(loss, grapher, epoch, prefix='train'):
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k:
            key_name = k
            value = v.item() if not isinstance(v, (float, np.float32,
                                                   np.float64)) else v
            grapher.register_single(
                {'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                plot_type='line')


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names)
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.clamp(images.detach(), 0.0, 1.0)
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')


def _add_loss_map(loss_tm1, loss_t):
    if not loss_tm1:  # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k:
                resultant[k] = v.detach()

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k:
            resultant[k] = loss_tm1[k] + v.detach()

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    for k in loss_map.keys():
        loss_map[k] /= loss_map['count']

    return loss_map


def train(epoch, model, gen_optimizer, inf_optimizer, loader, grapher):
    ''' train loop helper '''
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         grapher=grapher,
                         gen_optimizer=gen_optimizer,
                         inf_optimizer=inf_optimizer,
                         prefix='train')


def execute_graph(epoch,
                  model,
                  data_loader,
                  grapher,
                  gen_optimizer=None,
                  inf_optimizer=None,
                  prefix='test'):
    loss_map, num_samples = {}, 0

    for y1, y2 in data_loader:

        gen_optimizer.zero_grad()

        if args.mode == 'scratch':
            loss, loss_z, loss_y = model.wake_student({'y': [y1, y2]})
            loss_map['wake_loss_z_mean'] = loss_z
            loss_map['wake_loss_y_mean'] = loss_y
        elif args.mode == 'local':
            loss = model.distill_local()
        elif args.mode == 'our':
            loss = model.distill_our()

        loss.backward()
        loss_map['gen_param_norm_mean'] = torch.norm(
            nn.utils.parameters_to_vector(model.student.gennet.parameters()))
        loss_map['gen_grad_norm_mean'] = torch.norm(
            parameters_grad_to_vector(model.student.gennet.parameters()))
        loss_map['wake_loss_mean'] = loss

        gen_optimizer.step()

        if args.mode == 'scratch':
            inf_optimizer.zero_grad()
            loss = model.sleep_student()

            loss.backward()
            loss_map['inf_param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(
                    model.student.infnet.parameters()))
            loss_map['inf_grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.infnet.parameters()))
            loss_map['sleep_loss_mean'] = loss

            inf_optimizer.step()

    print('{}[Epoch {}][{} samples]: losses {}'.format(prefix, epoch,
                                                       num_samples,
                                                       str(loss_map)),
          flush=True)
    # plot the test accuracy, loss and images
    if grapher:  # only if grapher is not None
        register_plots(loss_map, grapher, epoch=epoch, prefix=prefix)
        grapher.show()

    loss_map.clear()

    return


def generate(HM, loader, fname):
    # sns.set_theme(style="darkgrid")
    sns.set_style("whitegrid")

    context = sns.plotting_context()
    context.update({
        'xtick.labelsize': 14,
        'axes.labelsize': 14.0,
        'font.size': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (5, 4)
    })
    sns.set_context(context)

    y1, y2 = loader.__iter__().__next__()
    y1 = y1.squeeze().cpu().numpy() * Y1_STD + Y1_MEAN
    y2 = y2.squeeze().cpu().numpy() * Y2_STD + Y2_MEAN
    cmap = sns.cubehelix_palette(start=0, light=0.9, as_cmap=True)

    sns.kdeplot(data={
        'duration': y1,
        'waiting': y2
    },
                x='duration',
                y='waiting',
                cmap=cmap,
                fill=True,
                clip=((0.5, 6), (30, 105)),
                levels=10,
                legend=False)

    plt.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.95)

    plt.savefig('data.pdf')
    plt.cla()

    param = HM.generate()
    y1, y2 = param['y']
    y1 = y1.squeeze().detach().cpu().numpy() * Y1_STD + Y1_MEAN
    y2 = y2.squeeze().detach().cpu().numpy() * Y2_STD + Y2_MEAN

    if args.mode == 'scratch':
        s = 0.5
    elif args.mode == 'our':
        s = 1
    elif args.mode == 'local':
        s = 1.5
    s = 0

    cmap = sns.cubehelix_palette(start=s, light=0.9, as_cmap=True)

    sns.kdeplot(data={
        'duration': y1,
        'waiting': y2
    },
                x='duration',
                y='waiting',
                cmap=cmap,
                fill=True,
                clip=((0.5, 6), (30, 105)),
                levels=10,
                legend=False)
    plt.savefig(fname + '.pdf')
    plt.cla()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    loader = DataLoader(OFDataset(), batch_size=args.batch_size, shuffle=False)

    if args.mode == 'scratch':
        student = HM(2, kwargs=vars(args))
        teacher = None
    elif args.mode in ('our', 'local'):
        student = HM(1, kwargs=vars(args))
        teacher = HM(2, kwargs=vars(args))

    student_teacher = StudentTeacher(teacher, student, kwargs=vars(args))

    if args.mode in ('our', 'local'):
        teacher_fname = os.path.join(args.ckpt_dir,
                                     'teacher-1000.pth.tar')
        print("loading teacher model {}".format(teacher_fname), flush=True)
        student_teacher.teacher.load_state_dict(torch.load(teacher_fname),
                                                strict=True)

    if args.resume != 0:
        load_fname = os.path.join(
            args.ckpt_dir, '{}-{}.pth.tar'.format(args.uid, str(args.resume)))
        print("loading eval model {}".format(load_fname), flush=True)
        student_teacher.student.load_state_dict(torch.load(load_fname),
                                                strict=True)

    # build the grapher object
    grapher = Grapher(env=args.uid,
                      server=args.visdom_url,
                      port=args.visdom_port)
    grapher = None

    return [student_teacher, loader, grapher]


def set_seed(seed):
    if seed is None:
        raise NotImplementedError('seed must be specified')
    print("setting seed %d" % args.seed, flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs.select(args.gpu_wait)

    # handle randomness / non-randomness
    set_seed(args.seed)

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loader, grapher = get_model_and_loader()

    # eval mode
    if args.eval:
        fname = args.uid + '-' + str(args.resume)
        generate(model.student, data_loader, fname)

    # train mode
    else:
        gen_optimizer = build_optimizer(model.student.gennet)
        inf_optimizer = build_optimizer(model.student.infnet)
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, 99999, 0.1)
        inf_scheduler = optim.lr_scheduler.StepLR(inf_optimizer, 99999, 0.1)

        print(
            "there are {} params with {} elems in the st-model and {} params in the student with {} elems"
            .format(len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.gennet.parameters())),
                    number_of_parameters(model.student.gennet)),
            flush=True)

        for epoch in range(args.resume + 1, args.epochs + 1):
            train(epoch, model, gen_optimizer, inf_optimizer, data_loader,
                  grapher)
            gen_scheduler.step()
            inf_scheduler.step()
            if epoch % 1000 == 0:
                fname = args.uid + '-' + str(epoch)
                generate(model.student, data_loader, fname)
                save_fname = os.path.join(args.ckpt_dir,
                                          '{}.pth.tar'.format(fname))
                print("saving model {}...".format(save_fname), flush=True)
                torch.save(model.student.state_dict(), save_fname)

    if grapher is not None:
        # dump config to visdom
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(
            model.student.config),
                         opts=dict(title="config"))

        grapher.save()  # save the remote visdom graphs


if __name__ == "__main__":
    run(args)
