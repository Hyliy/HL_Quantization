import numpy as np
import pytorch_lightning as pl
import torch
from cifar10_models.resnet import resnet18
from optimizer import ISTA
from torch.nn import functional as F


class Resnet(pl.LightningModule):
    def __init__(self, params={}):
        super().__init__()
        self.params = params
        self.model = resnet18()
        self.cmodel = resnet18()

        state_dict = torch.load('./cifar10_models/state_dicts/resnet18.pt')
        self.model.load_state_dict(state_dict)

        if params['method'] == 'QAT' or params['method'] == 'ISTA':
            self.automatic_optimization = False

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        epsilon = self.params['epsilon']
        lam = self.params['lam'] if 'lam ' in self.params else 1
        opt = self.optimizers()

        if self.params['method'] == 'QAT' or self.params['method'] == 'ISTA':
            torch.set_grad_enabled(False)
            cps, ps = [*self.cmodel.parameters()], [*self.model.parameters()]
            for i, (cp, qp) in enumerate(zip(cps, ps)):
                if 0 < i < (len(cps) - 2): # do not quantize the first and the last layer
                    qp.copy_((cp / epsilon).round() * epsilon)
                else:
                    qp.copy_(cp)
            torch.set_grad_enabled(True)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if self.params['method'] == 'QAT' or self.params['method'] == 'ISTA':
            self.model.zero_grad()
            self.manual_backward(loss)
            for cp, qp in zip(self.cmodel.parameters(), self.model.parameters()):
                cp.grad = qp.grad

            if self.params['method'] == 'QAT':
                opt.step()
            elif self.params['method'] == 'ISTA':
                opt.step_(self.model, x, y)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)

        torch.set_grad_enabled(False)
        hep = epsilon / 2
        sqep = epsilon ** 2 / 4
        penalty = 0.0
        for i, param in enumerate(self.model.parameters()):
            if 0 < i < (len(cps) - 2):
                penalty += (lam * (sqep - ((param % epsilon) - hep) ** 2)).sum()
        torch.set_grad_enabled(True)

        return {'loss': loss, 'acc': acc, 'penalty': penalty}

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar("Loss v.s. Step", outputs['loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Penalty v.s. Step", outputs['penalty'].item(), self.global_step)
        return outputs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_penalty = torch.stack([x['penalty'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss v.s. Epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Penalty v.s. Epoch", avg_penalty, self.current_epoch)
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
        optimizer = None
        if self.params['method'] == 'QAT':
            optimizer = torch.optim.Adam(self.cmodel.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        elif self.params['method'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lr'], momentum=.1, dampening=.2, weight_decay=self.params['weight_decay'])
        elif self.params['method'] == 'ISTA':
            optimizer = ISTA(self.cmodel.parameters(), self.params)
        return optimizer
