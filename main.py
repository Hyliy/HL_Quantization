import numpy as np
import pytorch_lightning as pl

import torch
import os
import time
import pickle
import regex
import re

from torch import Tensor
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from itertools import product
from model import Resnet
from pprint import pprint


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_params(f: str) -> dict:
    pattern = r'model(-[A-Za-z-_]*=[\w\.\d\(, \)]*)+(-eps=1e-08)(-[A-Za-z-_]*=[\w\.\d\(, \)]*)+.ckpt'
    reg = regex.compile(pattern=pattern)
    params = {}
    m = reg.match(f)
    if m:
        captures = [captures for i in range(1, 4) for captures in m.captures(i)]
        for capture in captures:
            k, v = re.sub(r'-', '', capture, 1).split('=')
            params[k] = float(v) if isfloat(v) else v
        params['betas'] = tuple(map(float, regex.sub(r'[\(\) ]', '', params['betas']).split(',')))
    return params


if __name__ == '__main__':
    img_preprocs_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        ]
    )
    img_preprocs_valid = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        ]
    )
    # train_imgs = datasets.CIFAR10(root='./data', train=True, download=True, transform=img_preprocs_train)
    # valid_imgs = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_preprocs_valid)

    train_imgs = datasets.CIFAR100(root='./data', train=True, download=False, transform=img_preprocs_train)
    valid_imgs = datasets.CIFAR100(root='./data', train=False, download=False, transform=img_preprocs_valid)

    logger_folder = 'logs-resnet20-cifar100'
    model_folder = 'model-resnet20-cifar100'
    batch_size = 128
    methods = ['ISTA', 'QAT']
    epochs = [200]
    epsilons = reversed([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128])
    # epsilons = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]

    for i, (max_epochs, epsilon, method) in enumerate(product(epochs, epsilons, methods)):
        params_QAT = dict(
            batch=batch_size,
            max_epochs=max_epochs,
            method='QAT',
            lr=3e-4,
            epsilon=epsilon
        )
        params_ISTA = dict(
            batch=batch_size,
            max_epochs=max_epochs,
            method='ISTA',
            L=1 / 3e-4,
            lam=0.01,
            epsilon=epsilon
        )
        model = None
        if method == 'QAT':
            model = Resnet(params_QAT)
        elif method == 'ISTA':
            model = Resnet(params_ISTA)
        name_QAT = '{params}'.format(
            params='-'.join([k + '={}'.format(v) if k != 'L' and k != 'epsilon' else k + '={}'.format(1 / v) for k, v in params_QAT.items()])
        )
        name_ISTA = '{params}'.format(
            params='-'.join([k + '={}'.format(v) if k != 'L' and k != 'epsilon' else k + '={}'.format(1 / v) for k, v in params_ISTA.items()])
        )
        name = None
        if method == 'QAT':
            name = name_QAT
        elif method == 'ISTA':
            name = name_ISTA

        train_loader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_imgs, batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

        logger = TensorBoardLogger(logger_folder, name=name)
        ckpt = ModelCheckpoint(
            monitor='acc',
            dirpath=model_folder,
            filename='{epoch}-{acc:.4f}',
            save_top_k=1,
            mode='max',
            prefix='model-{name}'.format(name=name)
        )

        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, logger=logger, callbacks=[ckpt])
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

    print('done')
