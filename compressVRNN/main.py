from __future__ import division
import os
import argparse
import pprint
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import fnmatch
from lxml import etree
from numpy.core.fromnumeric import repeat
import seaborn as sns
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence

from models.vae.VRNN import VRNN
from models.student_teacher import StudentTeacher
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
                    default=6,
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
                    default=1e-3,
                    metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--dzlambda', type=float, default=1e-3, help='dzlambda')
parser.add_argument('--STD', type=float, default=1, metavar='LR', help='STD')

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


def fetch_iamondb(data_path='data'):
    '''
    strokes_path = os.path.join(data_path, "lineStrokes")
    ascii_path = os.path.join(data_path, "ascii")
    train_files_path = os.path.join(data_path, "train.txt")
    valid_files_path = os.path.join(data_path, "valid.txt")

    if not os.path.exists(strokes_path) or not os.path.exists(ascii_path):
        raise ValueError("You must download the data from IAMOnDB, and"
                         "unpack in %s" % data_path)

    if not os.path.exists(train_files_path) or not os.path.exists(
            valid_files_path):
        raise ValueError("Cannot find concatenated train.txt and valid.txt"
                         "files! See the README in %s" % data_path)

    partial_path = data_path
    train_names = [
        f.strip() for f in open(train_files_path, mode='r').readlines()
    ]
    valid_names = [
        f.strip() for f in open(valid_files_path, mode='r').readlines()
    ]

    def construct_stroke_paths(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        files_path = os.path.join(strokes_path, primary_dir, sub_dir)

        #Dash is crucial to obtain correct match!
        files = fnmatch.filter(os.listdir(files_path), f + "-*.xml")
        files = [os.path.join(files_path, fi) for fi in files]
        files = sorted(
            files, key=lambda x: int(x.split(os.sep)[-1].split("-")[-1][:-4]))

        return files

    def construct_ascii_path(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        file_path = os.path.join(ascii_path, primary_dir, sub_dir, f + ".txt")

        return file_path

    def estimate_parameter(X):
        # X: list of numpy: N * n_points-1 * 2
        reshaped_X = X[0]
        for line in X[1:]:
            reshaped_X = np.concatenate([reshaped_X, line], axis=0)

        m, s = reshaped_X.mean(axis=0), reshaped_X.std(axis=0)
        return m, s

    train_stroke_files_s = [construct_stroke_paths(f) for f in train_names]
    valid_stroke_files_s = [construct_stroke_paths(f) for f in valid_names]

    train_ascii_files = [construct_ascii_path(f) for f in train_names]
    valid_stroke_files = [construct_stroke_paths(f) for f in valid_names]

    '''
    train_npy_x = os.path.join(data_path, "train_npy_x.npy")
    '''
    valid_npy_x = os.path.join(partial_path, "valid_npy_x.npy")
    train_npy_u = os.path.join(partial_path, "train_npy_u.npy")
    valid_npy_u = os.path.join(partial_path, "valid_npy_u.npy")

    train_set = (train_stroke_files_s, train_ascii_files, train_npy_x,
                 train_npy_u)

    valid_set = (valid_stroke_files_s, valid_stroke_files, valid_npy_x,
                 valid_npy_u)

    if 0:  #not os.path.exists(train_npy_x) or not os.path.exists(train_npy_u):
        for stroke_files_s, ascii_files, npy_x, npy_u in [
                train_set, valid_set
        ]:
            # list of nparray: N * n_points * 3
            x_set = []
            u_set = []

            for n in range(len(stroke_files_s)):
                if n % 100 == 0:
                    print("Processing file %i of %i" %
                          (n, len(stroke_files_s)))

                stroke_files = stroke_files_s[n]
                ascii_file = ascii_files[n]

                with open(ascii_file) as fp:
                    cleaned = [
                        t.strip() for t in fp.readlines()
                        if t != '\r\n' and t != '\n' and t != ' \r\n'
                    ]
                    cleaned = [l for l in cleaned if len(l) != 0]

                    # Try using CSR
                    idx = [n for n, li in enumerate(cleaned)
                           if li == "CSR:"][0]
                    cleaned_sub = cleaned[idx + 1:]
                    corrected_sub = []

                    for li in cleaned_sub:
                        # Handle edge case with %%%%% meaning new line?
                        if "%" in li:
                            print(li)
                            li2 = re.sub('\%\%+', '%', li).split("%")
                            li2 = [l.strip() for l in li2]
                            print(li2)
                            corrected_sub.extend(li2)
                        else:
                            corrected_sub.append(li)

                n_one_hot = 57
                u = [
                    np.zeros((len(li), n_one_hot), dtype='int16')
                    for li in corrected_sub
                ]

                # A-Z, a-z, space, apostrophe, comma, period
                charset = list(range(65, 90 + 1)) + list(range(
                    97, 122 + 1)) + [32, 39, 44, 46]
                tmap = {k: n + 1 for n, k in enumerate(charset)}

                # 0 for UNK/other
                tmap[0] = 0

                def tokenize_ind(line):

                    t = [ord(c) if ord(c) in charset else 0 for c in line]
                    r = [tmap[i] for i in t]

                    return r

                for n, li in enumerate(corrected_sub):
                    u[n][np.arange(len(li)), tokenize_ind(li)] = 1

                x = []

                for stroke_file in stroke_files:
                    with open(stroke_file) as fp:
                        tree = etree.parse(fp)
                        root = tree.getroot()
                        # Get all the values from the XML
                        # 0th index is stroke ID, will become up/down
                        s = np.array([[
                            i,
                            int(Point.attrib['x']),
                            int(Point.attrib['y'])
                        ] for StrokeSet in root
                                      for i, Stroke in enumerate(StrokeSet)
                                      for Point in Stroke])

                        # flip y axis
                        s[:, 2] = -s[:, 2]

                        # Get end of stroke points
                        c = s[1:, 0] != s[:-1, 0]
                        ci = np.where(c == True)[0]
                        nci = np.where(c == False)[0]

                        # set pen down
                        s[0, 0] = 0
                        s[nci, 0] = 0

                        # set pen up
                        s[ci, 0] = 1
                        s[-1, 0] = 1
                        x.append(s)

                x_set.extend(x)
                u_set.extend(u)

            raw_X = x_set
            raw_X0 = []
            raw_new_X = []

            for item in raw_X:
                raw_X0.append(item[1:, 0])
                raw_new_X.append(item[1:, 1:] - item[:-1, 1:])

            # X_mean, X_std = estimate_parameter(raw_new_X)
            # print(X_mean, X_std)
            X_mean = np.array([8.17868533, -0.11164117])
            X_std = np.array([41.95389001, 37.123557])

            new_x = []

            for n in range(len(raw_new_X)):
                normalized_value = (raw_new_X[n] - X_mean) / X_std
                new_x.append(
                    np.concatenate((raw_X0[n][:, None], normalized_value),
                                   axis=-1).astype('float32'))

            pickle.dump(new_x, open(npy_x, mode="wb"))
            pickle.dump(u_set, open(npy_u, mode='wb'))

    '''
    train_x = pickle.load(open(train_npy_x, mode="rb"))
    # valid_x = pickle.load(open(valid_npy_x, mode="rb"))
    # train_u = pickle.load(open(train_npy_u, mode="rb"))
    # print(len(train_u), len(train_x))

    return train_x, None


class IAMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x, self.u = fetch_iamondb()
        # N * n_points_in_line * 3
        # N * n_ascii * 57

        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError

        x = self.x[idx]
        # u = self.u[idx]

        return torch.tensor(x).cuda()  # , torch.tensor(u).cuda()


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
                         prefix='train')


def execute_graph(epoch,
                  model,
                  data_loader,
                  grapher,
                  optimizer=None,
                  prefix='test'):

    loss_t, loss_map, num_samples = {}, {}, 0

    repeat = 1

    for i, (x, lengths) in enumerate(data_loader):

        model.student.train()

        if i % repeat == 0:
            optimizer.zero_grad()

        if args.mode == 'scratch':
            loss, loss_t = model.forward_student(x, lengths)
        elif args.mode == 'local':
            loss, loss_t = model.distill_local()
        elif args.mode == 'our':
            loss, loss_t = model.distill_our()

        loss_repeat = loss / repeat
        loss_repeat.backward()

        if i % repeat == repeat - 1:
            loss_t['loss_mean'] = loss
            loss_t['param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(model.student.parameters()))
            loss_t['grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.parameters()))
            optimizer.step()

            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += x.shape[1] * repeat

            print('{}[Epoch {}][{} samples]: loss {}'.format(
                prefix, epoch, num_samples, loss.detach()),
                  flush=True)

            if num_samples % 320 == 0:
                fname = './IMAGE/{}-{}-{}'.format(args.uid, epoch, num_samples)
                generate(model.student, fname)

    loss_map = _mean_map(loss_map)  # reduce the map to get actual means
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


def plot_lines_iamondb_example(X, x0):

    X_mean = np.array([8.17868533, -0.11164117])
    X_std = np.array([41.95389001, 37.123557])

    X = np.concatenate([x0, X], axis=0)

    for i in range(1, X.shape[0]):
        X[i, 1:] = X[i - 1, 1:] + X[i, 1:] * X_std + X_mean

    non_contiguous = np.where(X[:, 0] == 1)[0] + 1
    # places where new stroke begin
    start = 0

    for end in non_contiguous:
        plt.plot(X[start:end, 1], X[start:end, 2])
        start = end

    plt.plot(X[start:, 1], X[start:, 2])


def generate(VRNN, fname):
    VRNN.eval()

    for vars in [(1.0, 0.0, 1.0), (0.8, 0.0, 1.0)]:

        scale_z, scale_x, scale_x_binary = vars
        param = VRNN.generate(scale_z, scale_x, scale_x_binary)
        lines = param['x'].detach().cpu().numpy()

        for index in range(0, 28, 5):
            plt.figure(figsize=(5, 4))
            plt.axis('equal')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            for i in range(5):
                plot_lines_iamondb_example(
                    lines[:, i + index, :], np.array([[0, 500,
                                                       1000 * i + 500]]))
            plt.savefig('{}-{}-{}-{}-{}.pdf'.format(fname, index, scale_z,
                                                    scale_x, scale_x_binary))
            plt.cla()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    dataset = IAMDataset()

    def collate_fn(train_data):
        train_data.sort(key=lambda data: len(data), reverse=True)
        data_length = [len(data) for data in train_data]
        train_data = pad_sequence(train_data,
                                  batch_first=False,
                                  padding_value=0)
        return train_data, data_length

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        collate_fn=collate_fn)

    if args.mode == 'scratch':
        student = VRNN(2, kwargs=vars(args))
        teacher = None
    elif args.mode in ('our', 'local'):
        student = VRNN(1, kwargs=vars(args))
        teacher = VRNN(2, kwargs=vars(args))

    student_teacher = StudentTeacher(teacher, student, kwargs=vars(args))

    if args.mode in ('our', 'local'):
        teacher_fname = os.path.join(args.ckpt_dir, 'teacher-6.pth.tar')
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
    # grapher = None

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
        fname = './IMAGE/{}-{}-test'.format(args.uid, args.resume)
        generate(model.student, fname)

    # train mode
    else:
        optimizer = build_optimizer(model.student)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 9999, 0.1)
        print(
            "there are {} params with {} elems in the st-model and {} params in the student with {} elems"
            .format(len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.parameters())),
                    number_of_parameters(model.student)),
            flush=True)

        for epoch in range(1, args.resume + 1):
            scheduler.step()

        for epoch in range(args.resume + 1, args.epochs + 1):
            train(epoch, model, optimizer, data_loader, grapher)
            scheduler.step()

            # if epoch % 50 == 0:
            save_fname = os.path.join(args.ckpt_dir,
                                      '{}-{}.pth.tar'.format(args.uid, epoch))
            torch.save(model.student.state_dict(), save_fname)
            save_fname = os.path.join(
                args.ckpt_dir, '{}-{}-optim.pth.tar'.format(args.uid, epoch))
            torch.save(optimizer.state_dict(), save_fname)

    if grapher is not None:
        # dump config to visdom
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(
            model.student.config),
                         opts=dict(title="config"))

        grapher.save()  # save the remote visdom graphs


if __name__ == "__main__":
    run(args)
