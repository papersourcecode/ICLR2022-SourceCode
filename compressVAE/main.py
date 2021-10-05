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
from datasets.loader import get_loader
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.fid import train_fid_model
from helpers.metrics import calculate_fid
from helpers.utils import float_type, ones_like, \
    append_to_csv, num_samples_in_loader, check_or_create_dir, \
    dummy_context, number_of_parameters
from helpers.distributions import set_decoder_std
import GPUs

parser = argparse.ArgumentParser(description='VAE compression')

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
parser.add_argument('--epochs',
                    type=int,
                    default=200,
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
parser.add_argument('--t-e-arch', type=int, default=1, help='teacher arch')
parser.add_argument('--t-d-arch', type=int, default=1, help='teacher arch')
parser.add_argument('--s-e-arch', type=int, default=1, help='teacher arch')
parser.add_argument('--s-d-arch', type=int, default=1, help='teacher arch')
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
                    default=1.0,
                    help='hyperparameter to scale KL term in ELBO')

parser.add_argument('--warmup-epoch', type=int, default=1, help='warmup epoch')
parser.add_argument(
    '--generative-scale-var',
    type=float,
    default=1.0,
    help='scale variance of prior in order to capture outliers')

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
parser.add_argument('--distill-KL',
                    type=int,
                    default=0,
                    help='use KL instead of WS22')
parser.add_argument('--mode',
                    type=str,
                    default='our',
                    help='retrain / replay / our')

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
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](model.parameters(),
                                                     lr=args.lr,
                                                     weight_decay=1e-4)


def build_dec_optimizer(model):
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
    return optim_map[args.optimizer.lower().strip()](chain(
        model.d0.parameters(),
        model.d1.parameters(),
        model.d2.parameters(),
        model.d3.parameters(),
        model.d4.parameters(),
    ),
                                                     lr=args.lr,
                                                     weight_decay=1e-4)


def build_enc_optimizer(model):
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
    return optim_map[args.optimizer.lower().strip()](chain(
        model.u1.parameters(),
        model.u2.parameters(),
        model.u3.parameters(),
        model.u4.parameters(),
        model.u5.parameters(),
    ),
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
            # WARM UP
            beta = min((epoch / args.warmup_epoch), 1.0) * args.beta
        else:
            beta = args.beta

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            # run the VAE and extract loss
            # case 1: forward: test / scratch / retrain
            if 'test' in prefix or model.teacher is None or args.mode == 'retrain':
                data = data.cuda() if args.cuda else data
                output_map = model(data)
            # case 2: generative replay: replay
            elif args.mode == 'replay':
                output_map = model.generative_replay()
            # case 3: distill: our
            elif args.mode == 'our':
                output_map = model.distill()

            loss_t = model.loss_function(output_map, beta)

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
        register_images(images, img_names, grapher, prefix=prefix)
        generate(model.student, grapher)  # generate samples
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


def generate(vae, grapher):

    vae.eval()
    # random generation
    _, gen, _ = vae.generate_synthetic_samples(args.batch_size)
    gen = torch.clamp(gen, 0, 1)
    grapher.register_single({'generated_samples': gen}, plot_type='imgs')
    grapher.show()


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

    s_vae = LVAE(loader.img_shp,
                 args.s_e_arch,
                 args.s_d_arch,
                 kwargs=vars(args))

    # resume
    if args.resume_model is not None:
        print("loading resume model {}".format(args.resume_model), flush=True)
        s_vae.load_state_dict(torch.load(args.resume_model), strict=True)

    t_vae = None

    # eval mode
    if args.eval_model is not None:
        print("loading eval model {}".format(args.eval_model), flush=True)
        s_vae.load_state_dict(torch.load(args.eval_model), strict=True)

    # train mode
    else:
        # with a teacher
        if args.teacher_model is not None:
            t_vae = LVAE(loader.img_shp,
                         args.t_e_arch,
                         args.t_d_arch,
                         kwargs=vars(args))
            print("loading teacher model {}".format(args.teacher_model),
                  flush=True)

            # init t_vae
            t_vae.load_state_dict(torch.load(args.teacher_model), strict=True)
            t_vae.requires_grad_(False)

            if args.mode == 'our':
                s_vae.u1.requires_grad_(False)
                s_vae.u2.requires_grad_(False)
                s_vae.u3.requires_grad_(False)
                s_vae.u4.requires_grad_(False)
                s_vae.u5.requires_grad_(False)

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(t_vae, s_vae, kwargs=vars(args))

    # build the grapher object
    grapher = Grapher(env=args.uid,
                      server=args.visdom_url,
                      port=args.visdom_port)

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


def transfer_beta_sigma():
    sigma = np.sqrt(args.beta / 2)
    args.beta = 1.0
    set_decoder_std(sigma)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs.select(args.gpu_wait)

    # handle randomness / non-randomness
    set_seed(args.seed)

    transfer_beta_sigma()

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loader, grapher = get_model_and_loader()

    # build a classifier to use for FID
    fid_model = None
    if args.calculate_fid_with is not None:
        fid_batch_size = args.batch_size if args.calculate_fid_with == 'conv' else 256
        fid_model = train_fid_model(args, args.calculate_fid_with,
                                    fid_batch_size)

    # eval mode
    if args.eval_model is not None:
        print("evaluating model {}...".format(args.eval_model), flush=True)
        test(0, model, data_loader.test_loader, grapher, fid_model)

    # train mode
    else:
        optimizer = build_optimizer(model.student)  # collect our optimizer

        print(
            "there are {} params with {} elems in the st-model and {} params in the student with {} elems"
            .format(len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.parameters())),
                    number_of_parameters(model.student)),
            flush=True)
        start_epoch = 1
        if args.resume_model is not None:
            start_epoch = args.resume_epoch
        for epoch in range(start_epoch, args.epochs + 1):
            train(epoch, model, optimizer, data_loader.train_loader, grapher)

        test(0, model, data_loader.test_loader, grapher, fid_model)

        save_fname = os.path.join(args.ckpt_dir, args.uid + '-model.pth.tar')
        print("saving model {}...".format(save_fname), flush=True)
        torch.save(model.student.state_dict(), save_fname)

    # dump config to visdom
    grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(
        model.student.config),
                     opts=dict(title="config"))

    grapher.save()  # save the remote visdom graphs


if __name__ == "__main__":
    run(args)
