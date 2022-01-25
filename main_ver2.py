import re
from itertools import product

import pytorch_lightning as pl
import regex
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets

from model_liu_ver2 import Resnet


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
            # T.RandomCrop(32, padding=4),
            # T.RandomHorizontalFlip(),
            # T.Resize(224, interpolation=3),
            T.Resize(32),
            T.ToTensor(),
            # T.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        ]
    )
    img_preprocs_valid = T.Compose(
        [
            T.Resize(32),
            T.ToTensor(),
            # T.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        ]
    )

    # train_imgs = datasets.CIFAR10(root='./data', train=True, download=False, transform=img_preprocs_train)
    # valid_imgs = datasets.CIFAR10(root='./data', train=False, download=False, transform=img_preprocs_valid)
    # train_imgs = datasets.CIFAR100(root='./data', train=True, download=True, transform=img_preprocs_train)
    # valid_imgs = datasets.CIFAR100(root='./data', train=False, download=True, transform=img_preprocs_valid)
    train_imgs = datasets.MNIST(root='./data', train=True, download=True, transform=img_preprocs_train)
    valid_imgs = datasets.MNIST(root='./data', train=False, download=True, transform=img_preprocs_valid)

    logger_folder = 'logs-resnet50-mnist-sgd-final-quantized-all-redo'
    model_folder = 'model-resnet50-mnist-sgd-final-quantized-all-redo'
    batch_size = 256
    methods = ['ISTA_LIU']
    epochs = [200]
    epsilons = [1 / 4]

    for i, (max_epochs, epsilon, method) in enumerate(product(epochs, epsilons, methods)):
        params_QAT = dict(
            batch=batch_size,
            max_epochs=max_epochs,
            method='QAT',
            lr=3e-4,
            L=1 / 3e-4,
            lam=1e-3,
            epsilon=epsilon
        )
        params_ISTA = dict(
            batch=batch_size,
            max_epochs=max_epochs,
            method='ISTA_LIU',
            L=1 / 5e-3,
            lam=1e-4,
            epsilon=epsilon
        )
        model = None
        if method == 'QAT':
            # model = Resnet(params_QAT)
            model = Resnet.load_from_checkpoint(
                'model-resnet18-mnist-sgd-final-no-pre-trained/model-resnet50-mnist-redo-acc=0.8165.ckpt',
                params=params_QAT
            )
        elif method == 'ISTA_LIU':
            # model = Resnet(params_ISTA)
            model = Resnet.load_from_checkpoint(
                'model-resnet18-mnist-sgd-final-no-pre-trained/model-resnet50-mnist-redo-acc=0.8165.ckpt',
                params=params_ISTA
            )
        name_QAT = '{params}'.format(
            params='-'.join([k + '={}'.format(v) if k != 'L' and k != 'epsilon' else k + '={}'.format(1 / v) for k, v in params_QAT.items()])
        )
        name_ISTA = '{params}'.format(
            params='-'.join([k + '={}'.format(v) if k != 'L' and k != 'epsilon' else k + '={}'.format(1 / v) for k, v in params_ISTA.items()])
        )
        name = None
        if method == 'QAT':
            name = name_QAT
        elif method == 'ISTA_LIU':
            name = name_ISTA

        train_loader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_imgs, batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

        logger = TensorBoardLogger(logger_folder, name=name)
        ckpt = ModelCheckpoint(
            monitor='acc',
            dirpath=model_folder,
            filename='model-{name}-'.format(name=name) + '{epoch}-{acc:.4f}',
            save_top_k=1,
            mode='max'
        )

        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, logger=logger, callbacks=[ckpt])
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

    print('done')
