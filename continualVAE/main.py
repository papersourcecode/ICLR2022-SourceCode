import os
import argparse
import pprint
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
from itertools import chain

from models.vae.LVAE import LVAE
from models.student_teacher import StudentTeacher
from datasets.loader import get_loader, get_split_data_loaders
from helpers.grapher import Grapher
from helpers.fid import train_fid_model
from helpers.metrics import calculate_fid
from helpers.utils import float_type, ones_like, \
    append_to_csv, num_samples_in_loader, check_or_create_dir, \
    dummy_context, number_of_parameters
from helpers.distributions import set_decoder_std
import GPUs

parser = argparse.ArgumentParser(description='VAE distillation Pytorch')

# Task parameters
parser.add_argument(
    '--uid',
    type=str,
    default="",
    help="add a custom task-specific unique id (default: None)")
parser.add_argument(
    '--task',
    type=str,
    default="mnist",
    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)"""
)
parser.add_argument('--target-class', type=int, default=0)
parser.add_argument('--continual-step', type=int, default=0)
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    metavar='N',
                    help='minimum number of epochs to train (default: 10)')

parser.add_argument('--std-exp-activation',
                    action='store_true',
                    default=False,
                    help='use exp activation to derive std')

parser.add_argument('--download',
                    type=int,
                    default=1,
                    help='download dataset from s3 (default: 1)')
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

parser.add_argument(
    '--calculate-fid-with',
    type=str,
    default=None,
    help='enables FID calc & uses model conv/inceptionv3  (default: None)')
parser.add_argument(
    '--fid-model',
    type=str,
    default='./FID/tmp.pth.tar',
    help=
    'FID model you are to use, load if exists, retrain otherwise(default: None)'
)
parser.add_argument('--fid-interval',
                    type=int,
                    default=9999,
                    help='FID interval(default: 1)')

# train / eval or resume modes
parser.add_argument('--teacher-model',
                    type=str,
                    default=None,
                    help='teacher you are going to learn from (default: None)')
parser.add_argument('--eval-model',
                    type=str,
                    default=None,
                    help='model you are to eval (default: None)')
parser.add_argument('--resume-model',
                    type=str,
                    default=None,
                    help='model you are to resume (default: None)')
parser.add_argument('--resume-epoch',
                    type=int,
                    default=9999,
                    help='epoch to begin with')
# Model parameters
parser.add_argument('--nll-type',
                    type=str,
                    default='gaussian',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--t-e-arch', type=int, default=5, help='teacher arch')
parser.add_argument('--t-d-arch', type=int, default=5, help='teacher arch')
parser.add_argument('--s-e-arch', type=int, default=5, help='teacher arch')
parser.add_argument('--s-d-arch', type=int, default=5, help='teacher arch')
# Optimization related
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help="specify optimizer (default: SGD)")
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--beta',
                    type=float,
                    default=1e-1,
                    help='hyperparameter to scale KL term in ELBO')

parser.add_argument('--alpha', type=float, default=1.0, help='alpha')

parser.add_argument('--warmup-epoch',
                    type=int,
                    default=25,
                    help='warmup epoch')

parser.add_argument('--distill-z-kl-lambda',
                    type=float,
                    default=0.0,
                    help='distill z kl lambda (default: 0.0)')
parser.add_argument('--distill-z-reduction',
                    type=str,
                    default='mean',
                    help='how to reduce z')
parser.add_argument('--distill-share-z',
                    type=int,
                    default=0,
                    help='share z instead of noise')
parser.add_argument('--bayesian-update',
                    type=int,
                    default=0,
                    help='bayesian update')
parser.add_argument('--distill-KL',
                    type=int,
                    default=0,
                    help='use KL instead of WS22')

parser.add_argument('--mode', type=str, default='our', help='replay / our')

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


def train(epoch, model, optimizer, loader, grapher):
    ''' train loop helper '''
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         grapher=grapher,
                         optimizer=optimizer,
                         fid_model=None,
                         prefix='train')


def test(epoch, model, loader, grapher, fid_model):
    ''' test loop helper '''
    return execute_graph(epoch,
                         model=model,
                         data_loader=loader,
                         grapher=grapher,
                         fid_model=fid_model,
                         optimizer=None,
                         prefix='test')


def execute_graph(epoch,
                  model,
                  data_loader,
                  grapher,
                  fid_model=None,
                  optimizer=None,
                  prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    model.eval()
    if 'train' in prefix:
        model.student.train()

    assert optimizer is not None if 'train' in prefix else optimizer is None

    loss_map, num_samples = {}, 0

    for data, _ in data_loader:
        if 'train' in prefix:
            # zero gradients on optimizer
            # before forward pass
            optimizer.zero_grad()
            # WARM UP only in first step
            beta = args.beta
            if model.continual_step == 0:
                beta = min((epoch / args.warmup_epoch), 1.0) * args.beta
            alpha = args.alpha
        else:
            beta = args.beta
            alpha = args.alpha

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            # run the VAE and extract loss
            data = data.cuda() if args.cuda else data
            if model.continual_step == 0 or 'test' in prefix:
                output_map = model(data)
            elif args.mode == 'our':
                output_map = model.distill_forward(data)
            elif args.mode == 'replay':
                output_map = model.replay_forward(data)

            loss_t = model.loss_function(output_map, beta, alpha)

        if 'train' in prefix:
            # compute bp and optimize
            loss_t['loss_mean'].backward()
            loss_t['param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(model.student.parameters()))
            loss_t['grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.parameters()))
            optimizer.step()
            # print(loss_t['grad_norm_mean'], flush=True)

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += data.size(0)

    if epoch % args.fid_interval == 0:
        # FID
        if fid_model is not None:
            fid = calculate_fid(fid_model=fid_model,
                                model=model.student,
                                continual_step=model.continual_step,
                                loader=data_loader,
                                grapher=grapher,
                                num_samples=1000,
                                cuda=args.cuda)
            print('{}[Epoch {}][{} samples]: FID {}'.format(
                prefix, epoch, num_samples, str(fid)),
                  flush=True)

        # calculate_IS(model.student, data_loader)

    loss_map = _mean_map(loss_map)  # reduce the map to get actual means
    print('{}[Epoch {}][{} samples]: losses {}'.format(prefix, epoch,
                                                       num_samples,
                                                       str(loss_map)),
          flush=True)
    # plot the test accuracy, loss and images
    if grapher:  # only if grapher is not None
        register_plots(loss_map, grapher, epoch=epoch, prefix=prefix)

        images = []
        img_names = []
        if 'student' in output_map.keys() and 'x' in output_map.keys():
            images += [output_map['x'], output_map['student']['x_reconstr']]
            img_names += ['original_imgs', 'vae_reconstructions']
        if 'distill' in output_map.keys():
            images += [
                output_map['distill']['gen_teacher'],
                output_map['distill']['gen_student']
            ]
            img_names += ['generation_teacher', 'generation_student']
        if 'replay' in output_map.keys():
            images += [
                output_map['gen_teacher'], output_map['replay']['x_reconstr']
            ]
            img_names += ['generation_teacher', 'replay_reconstructions']
        register_images(images, img_names, grapher, prefix=prefix)
        generate(model.student, model.continual_step,
                 grapher)  # generate samples
        grapher.show()

    loss_map.clear()

    return


def calculate_IS(vae, data_loader):
    loss_map = {}
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.cuda() if args.cuda else data
            IS = vae.importance_sampling_prob(data, 5000)
            if IS == 999:
                return
            loss_map = _add_loss_map(loss_map, {'IS_mean': IS})
    loss_map = _mean_map(loss_map)
    print('importance sampling -log probability estimation: {}'.format(
        loss_map['IS_mean']))
    return


def calculate_recon_err(vae, data_loader):
    loss_map = {}
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.cuda() if args.cuda else data
            rec = vae.reconstruct(data)
            if args.nll_type == 'bernoulli':
                err = (rec != data).type(torch.float).mean()
            elif args.nll_type == 'gaussian':
                pass
            loss_map = _add_loss_map(loss_map, {'recon_err_mean': err})
    loss_map = _mean_map(loss_map)
    return loss_map['recon_err_mean']


NOISE_LIST = None


def generate(vae, continual_step, grapher):

    vae.eval()
    global NOISE_LIST
    for generating_step in range(continual_step + 1):
        # random generation
        _, gen, param = vae.generate_synthetic_samples(args.batch_size,
                                                       generating_step,
                                                       generating_step + 1,
                                                       noise_list=NOISE_LIST)
        NOISE_LIST = param['noise']
        gen = torch.clamp(gen, 0, 1)
        grapher.register_single(
            {
                'generated_samples_{}_{}'.format(continual_step, generating_step):
                gen
            },
            plot_type='imgs')
        grapher.show()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    loaders = get_split_data_loaders(args, args.target_class)

    print("train = ",
          num_samples_in_loader(loaders[-1].train_loader),
          " | test = ",
          num_samples_in_loader(loaders[-1].test_loader),
          " | shape ",
          loaders[-1].img_shp,
          flush=True)

    # append the image shape to the config & build the VAE
    args.img_shp = loaders[-1].img_shp

    s_vae = LVAE(args.img_shp, args.s_e_arch, args.s_d_arch, kwargs=vars(args))

    t_vae = None

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(t_vae, s_vae, kwargs=vars(args))

    # build the grapher object
    grapher = Grapher(env=args.uid,
                      server=args.visdom_url,
                      port=args.visdom_port)

    return student_teacher, loaders, grapher


def set_seed(seed):
    if seed is None:
        raise NotImplementedError('seed must be specified')
    print("setting seed %d" % args.seed, flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def transfer_beta_sigma():
    sigma = np.sqrt(args.beta / 2)
    args.beta = 1.0
    set_decoder_std(sigma)


def continual_loop(model, data_loaders, grapher, fid_model, continual_step):

    # eval mode
    if args.eval_model is not None:
        if continual_step == 0:
            eval_model_step = os.path.join(
                args.ckpt_dir,
                args.eval_model + '-model-{}.pth.tar'.format(continual_step))
        else:
            eval_model_step = os.path.join(
                args.ckpt_dir,
                args.eval_model + '-model-{}.pth.tar'.format(continual_step))
        print("evaluating model {}...".format(eval_model_step), flush=True)
        model.student.load_state_dict(torch.load(eval_model_step), strict=True)
        for test_c_s in range(continual_step + 1):
            model.continual_step = test_c_s
            test(0, model, data_loaders[test_c_s].test_loader, grapher,
                 fid_model)

    # train mode
    else:
        if continual_step != 0:
            model.student.load_state_dict(torch.load(
                os.path.join(
                    args.ckpt_dir, args.uid +
                    '-model-{}.pth.tar'.format(continual_step - 1))),
                                          strict=True)

            model.teacher = deepcopy(model.student)

        optimizer = build_optimizer(model.student)  # collect our optimizer

        print(
            "there are {} params with {} elems in the st-model and {} params in the student with {} elems"
            .format(len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.parameters())),
                    number_of_parameters(model.student)),
            flush=True)

        model.continual_step = continual_step

        real_epoch = args.epochs
        for epoch in range(1, real_epoch + 1):
            train(epoch, model, optimizer,
                  data_loaders[continual_step].train_loader, grapher)

        for test_c_s in range(continual_step + 1):
            model.continual_step = test_c_s
            test(0, model, data_loaders[test_c_s].test_loader, grapher,
                 fid_model)

        # save
        save_fname = os.path.join(
            args.ckpt_dir,
            args.uid + '-model-{}.pth.tar'.format(continual_step))
        print("saving model {}...".format(save_fname), flush=True)
        torch.save(model.student.state_dict(), save_fname)


def run(args):
    ratio_list = [
        0.11113579, 0.26698059, 0.512505, 0.20457159, 0.02244335, 0.15157528,
        0.24079586, 0.23453225, 0.23925093, 0.14799185, 0.05089857, 0.20519351,
        0.14216753, 0.05756692, 0.04668829, 0.06511878, 0.06276438, 0.04194986,
        0.38692195, 0.45503186, 0.41676909, 0.48342786, 0.04154512, 0.11514864,
        0.83493996, 0.28414257, 0.0429469, 0.27744461, 0.07977828, 0.06572096,
        0.05651064, 0.48208037, 0.20840182, 0.31956722, 0.18892492, 0.04846026,
        0.4724357, 0.12296704, 0.07271507, 0.77361685
    ]
    args.ratio = ratio_list[args.target_class]
    args.reverse = False
    if args.ratio > 0.5:
        args.reverse = True
        args.ratio = 1 - args.ratio

    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs.select(args.gpu_wait)

    # handle randomness / non-randomness
    set_seed(args.seed)

    transfer_beta_sigma()

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loaders, grapher = get_model_and_loader()

    # build a classifier to use for FID
    fid_model = None
    if args.calculate_fid_with is not None:
        fid_batch_size = args.batch_size if args.calculate_fid_with == 'conv' else 256
        fid_model = train_fid_model(args, args.calculate_fid_with,
                                    fid_batch_size)

    continual_loop(model, data_loaders, grapher, fid_model,
                   args.continual_step)

    # dump config to visdom
    grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(
        model.student.config),
                     opts=dict(title="config"))

    grapher.save()  # save the remote visdom graphs


if __name__ == "__main__":
    run(args)
