import numpy as np
import torch
from model_liu import Resnet


if __name__ == '__main__':
    params_ISTA = dict(
        batch=256,
        max_epochs=200,
        method='ISTA',
        L=1 / 3e-4,
        lam=1e-3,
        epsilon=1
    )
    model = Resnet.load_from_checkpoint('model-resnet18-mnist-sgd-final-quantized-all/model-batch=256-max_epochs=200-method=QAT-lr=0.0003-L=0.0003-lam=0.001-epsilon=4.0-epoch=180-acc=0.9615.ckpt'
                                        , params=params_ISTA
                                        )
    model.eval()
    with torch.no_grad():
        for n, p in model.model.named_parameters():
            p_ = torch.unique(p / (1 / 4))
            print(n, p_.size()[0], p_.max().item(), p_.min().item())

    print('done')