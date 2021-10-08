import os
import argparse
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.vib.VIB import VIB
from models.student_teacher import StudentTeacher
from datasets.loader import get_loader
from optimizers.adamnormgrad import AdamNormGrad
from helpers.utils import num_samples_in_loader, dummy_context
import GPUs

parser = argparse.ArgumentParser(description='compress VIB')

# Task parameters
parser.add_argument(
    '--uid',
    type=str,
    default="",
    help="add a custom task-specific unique id (default: None)")
parser.add_argument('--task', type=str, default="mnist", help="")
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='')

parser.add_argument('--download', type=int, default=1, help='')
parser.add_argument('--data-dir',
                    type=str,
                    default='../datasets-torchvision',
                    metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--ckpt-dir',
                    type=str,
                    default='./CKPT',
                    metavar='OD',
                    help='directory which contains ckpt')

# train / eval or resume modes
parser.add_argument('--teacher-model',
                    type=str,
                    default=None,
                    help='teacher you are going to learn from (default: None)')
# Model parameters

parser.add_argument('--t-arch', type=int, default=1, help='teacher arch')
parser.add_argument('--s-arch', type=int, default=1, help='teacher arch')
# Optimization related
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help="specify optimizer (default: SGD)")
parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--train-MCsamples',
                    type=int,
                    default=1,
                    help='MC samples in training')
parser.add_argument('--beta',
                    type=float,
                    default=0.01,
                    help='hyperparameter to scale KL term in IBs')

parser.add_argument('--distill-out-kl-lambda',
                    type=float,
                    default=0.0,
                    help='distill out kl lambda (default: 0.0)')
parser.add_argument('--distill-z-kl-lambda',
                    type=float,
                    default=0.0,
                    help='distill z kl lambda (default: 0.0)')

parser.add_argument('--temperature',
                    type=float,
                    default=1.0,
                    help='temperature')

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

    return optim_map[args.optimizer.lower().strip()](model.parameters(),
                                                     lr=args.lr,
                                                     weight_decay=1e-4)


def build_scheduler(optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, [80, 110, 140, 170],
                                          gamma=0.1)


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


def train(epoch, model, optimizer, loader):
    ''' train loop helper '''
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         optimizer=optimizer,
                         prefix='train')


def test(epoch, model, loader):
    ''' test loop helper '''
    return execute_graph(epoch,
                         model=model,
                         data_loader=loader,
                         optimizer=None,
                         prefix='test')


def execute_graph(epoch, model, data_loader, optimizer=None, prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    model.eval()
    if 'train' in prefix:
        model.student.train()

    assert optimizer is not None if 'train' in prefix else optimizer is None

    loss_map, num_samples = {}, 0

    for data, y in data_loader:
        if 'train' in prefix:
            # zero gradients on optimizer
            # before forward pass
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            # run the VAE and extract loss
            data = data.cuda() if args.cuda else data
            y = y.cuda() if args.cuda else y
            MCsamples = args.train_MCsamples if 'train' in prefix else 12
            output_map = model(data, MCsamples)
            loss_t = model.loss_function(y, output_map)

        if 'train' in prefix:
            # compute bp and optimize
            loss_t['loss_mean'].backward()
            loss_t['param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(model.student.parameters()))
            loss_t['grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.parameters()))
            optimizer.step()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += data.size(0)

    loss_map = _mean_map(loss_map)  # reduce the map to get actual means
    print('{}[Epoch {}][{} samples]: losses {}'.format(prefix, epoch,
                                                       num_samples,
                                                       str(loss_map)),
          flush=True)

    metric = loss_map['acc_mean']
    loss_map.clear()

    return float(metric)


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    loader = get_loader(args)

    print("train = ",
          num_samples_in_loader(loader.train_loader),
          " | test = ",
          num_samples_in_loader(loader.test_loader),
          " | shape ",
          loader.img_shp,
          flush=True)

    # append the image shape to the config & build the VAE
    args.img_shp = loader.img_shp,

    s_vae = VIB(loader.img_shp,
                loader.output_size,
                args.s_arch,
                kwargs=vars(args))

    t_vae = None

    # with a teacher
    if args.teacher_model is not None:
        t_vae = VIB(loader.img_shp,
                    loader.output_size,
                    args.t_arch,
                    kwargs=vars(args))
        print("loading teacher model {}".format(args.teacher_model),
              flush=True)

        # init t_vae
        t_vae.load_state_dict(torch.load(args.teacher_model), strict=True)

        t_vae.requires_grad_(False)

    student_teacher = StudentTeacher(t_vae, s_vae, kwargs=vars(args))

    return [student_teacher, loader]


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
    model, data_loader = get_model_and_loader()

    optimizer = build_optimizer(model.student)  # collect our optimizer
    scheduler = build_scheduler(optimizer)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, data_loader.train_loader)
        metric = test(epoch, model, data_loader.test_loader)
        scheduler.step()

    print("saving model with metric {}...".format(metric), flush=True)

    save_fname = os.path.join(args.ckpt_dir, args.uid + '-model.pth.tar')
    torch.save(model.student.state_dict(), save_fname)


if __name__ == "__main__":
    run(args)
