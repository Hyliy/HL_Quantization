import numpy as np
import pytorch_lightning as pl

import timm
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

if __name__ == '__main__':
    # model_names = timm.list_models()
    # pprint(model_names)
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

    logger_folder = 'logs-resnet18-pretrained-cifar100'
    model_folder = 'model-resnet18-pretrained-cifar100'
    batch_size = 256

    train_imgs = datasets.CIFAR100(root='./data', train=True, download=False, transform=img_preprocs_train)
    valid_imgs = datasets.CIFAR100(root='./data', train=False, download=False, transform=img_preprocs_valid)
    model = Resnet.load_from_checkpoint('./model-resnet18-pretrained-cifar100/model-batch=256-max_epochs=1000-method=Lookahead-epoch=607-acc=0.5370.ckpt', params=dict(method='SGD', epsilon=0))
    train_loader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_imgs, batch_size=batch_size, shuffle=False)

    name = 'ver1-batch={}-max_epochs=1000-method=Lookahead'.format(batch_size)
    logger = TensorBoardLogger(logger_folder, name=name)
    ckpt = ModelCheckpoint(
        monitor='acc',
        dirpath=model_folder,
        filename='{epoch}-{acc:.4f}',
        save_top_k=1,
        mode='max',
        prefix='model-{name}'.format(name=name)
    )

    trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=logger, callbacks=[ckpt])
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)