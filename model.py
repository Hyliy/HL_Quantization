import numpy as np
import pytorch_lightning as pl
import torch
import timm
import json
from optimizer import ISTA
from torch.nn import functional as F
from timm.optim.lookahead import Lookahead
from codebase.networks.natnet import NATNet


class Resnet(pl.LightningModule):
    '''Resnet model for testing'''

    def __init__(self, params={}):
        '''the params records the hyper-parameters of the model and don't confuse with the parameters for the optimizer which stands for the weights of the model.'''
        super().__init__()
        self.params = params
        config = json.load(open('./subnets/cifar100/net-img@224-flops@796-top1@88.3/net.config'))
        self.model = NATNet.build_from_config(config, pretrained=True)
        config = json.load(open('./subnets/cifar100/net-img@224-flops@796-top1@88.3/net.config'))
        self.cmodel = NATNet.build_from_config(config, pretrained=True)
        # self.model = timm.create_model('efficientnet_l2', pretrained=False)
        # self.cmodel = timm.create_model('resnet50', pretrained=False)
        # self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=False)  # the model that we try to quantize
        # self.model.load_state_dict(torch.load('./cifar100_resnet20-23dac2f1.pt'))
        # self.cmodel = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=False)  # the continuous model

        # self.model = timm.create_model('resnet18', pretrained=True)
        # self.cmodel = timm.create_model('resnet18', pretrained=False)
        # self.model = resnet18()
        # self.cmodel = resnet18()
        # state_dict = torch.load('./cifar10_models/state_dicts/resnet18.pt')
        # self.model.load_state_dict(state_dict)

        if params['method'] == 'QAT' or params['method'] == 'ISTA':  # if the optimization method is either QAT (benchmark) or ISAT (our method), then, we turn the auto optimization off
            self.automatic_optimization = False

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        '''Acquire the hyper-parameters from the params'''
        epsilon = self.params['epsilon'] if 'epsilon' in self.params else 1
        lam = self.params['lam'] if 'lam ' in self.params else 1
        opt = self.optimizers()

        if self.params['method'] == 'QAT' or self.params['method'] == 'ISTA':
            torch.set_grad_enabled(False)
            cps, ps = [*self.cmodel.parameters()], [*self.model.parameters()]
            for i, (cp, qp) in enumerate(zip(cps, ps)):
                if 0 < i < (len(cps) - 2):  # do not quantize the first and the last layer and copy the continuous gradient to the quantized one and do the quantization accordingly
                    qp.copy_((cp / epsilon).round() * epsilon)
                else:
                    qp.copy_(cp)
            torch.set_grad_enabled(True)
        '''Perform the forward pass and backdprop'''
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if self.params['method'] == 'QAT' or self.params['method'] == 'ISTA':
            self.model.zero_grad()
            self.manual_backward(loss)
            for cp, qp in zip(self.cmodel.parameters(), self.model.parameters()):
                cp.grad = qp.grad  # copy the gradient to the continuous model

            if self.params['method'] == 'QAT':
                opt.step()
            elif self.params['method'] == 'ISTA':
                opt.step_(self.model, x, y)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)

        return {'loss': loss, 'acc': acc}

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar("Loss v.s. Step", outputs['loss'].item(), self.global_step)
        # self.logger.experiment.add_scalar("Penalty v.s. Step", outputs['penalty'].item(), self.global_step)
        return outputs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_penalty = torch.stack([x['penalty'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss v.s. Epoch", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("Penalty v.s. Epoch", avg_penalty, self.current_epoch)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {'acc': acc}

    def validation_step_end(self, outputs):
        self.logger.experiment.add_scalar("Validation Acc v.s. Step", outputs['acc'], self.global_step)
        return outputs

    def validation_epoch_end(self, outputs):
        avg_acc = np.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation Acc v.s. Epoch", avg_acc, self.current_epoch)
        self.log('acc', avg_acc)
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {'acc': acc}

    def test_epoch_end(self, outputs):
        avg_acc = np.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Test Acc v.s. Epoch", avg_acc, self.current_epoch)
        return {'acc': avg_acc}

    def configure_optimizers(self):
        '''Setting the optimizer according to hyper-parameter'''
        optimizer = None
        if self.params['method'] == 'QAT':
            optimizer = torch.optim.Adam(self.cmodel.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
            optimizer = Lookahead(optimizer, alpha=.8, k=5)
        elif self.params['method'] == 'ISTA':
            optimizer = ISTA(self.cmodel.parameters(), self.params)
        return optimizer


class BitM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnetv2_101x3_bitm', pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        '''Perform the forward pass and backdprop'''
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {'loss': loss, 'acc': acc}

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar("Loss v.s. Step", outputs['loss'].item(), self.global_step)
        # self.logger.experiment.add_scalar("Penalty v.s. Step", outputs['penalty'].item(), self.global_step)
        return outputs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # avg_penalty = torch.stack([x['penalty'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss v.s. Epoch", avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar("Penalty v.s. Epoch", avg_penalty, self.current_epoch)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {'acc': acc}

    def validation_step_end(self, outputs):
        self.logger.experiment.add_scalar("Validation Acc v.s. Step", outputs['acc'], self.global_step)
        return outputs

    def validation_epoch_end(self, outputs):
        avg_acc = np.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation Acc v.s. Epoch", avg_acc, self.current_epoch)
        self.log('acc', avg_acc)
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {'acc': acc}

    def test_epoch_end(self, outputs):
        avg_acc = np.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Test Acc v.s. Epoch", avg_acc, self.current_epoch)
        return {'acc': avg_acc}

    def configure_optimizers(self):
        '''Setting the optimizer according to hyper-parameter'''
        optimizer = None
        if self.params['method'] == 'QAT':
            optimizer = torch.optim.Adam(self.cmodel.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
            optimizer = Lookahead(optimizer, alpha=.8, k=5)
        elif self.params['method'] == 'ISTA':
            optimizer = ISTA(self.cmodel.parameters(), self.params)
        return optimizer
