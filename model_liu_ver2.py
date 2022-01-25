import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.models as models
from torch.nn import functional as F
from optimizer_liu_ver2 import ISTA_LIU, QAT


class Resnet(pl.LightningModule):
    """Resnet model for testing"""

    def __init__(self, params={}):
        """the params records the hyper-parameters of the model and don't confuse with the parameters for the optimizer which stands for the weights of the model."""
        super().__init__()
        self.multiplier = 1
        self.old_parameters = None  # list of paramters from previous iteration
        self.params = params
        self.SoftThresholding_counter = 0

        self.model = models.resnet18(num_class=10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cmodel = models.resnet18(num_class=10)
        self.cmodel.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.set_epsilons_ = [params['epsilon'] for _ in range(len([*self.model.parameters()]))]
        self.names = [p[0] for p in self.model.named_parameters()]
        self.max_ = 1
        self.min_ = -1

        if (
                params["method"] == "QAT" or params["method"] == "ISTA_LIU"
        ):  # if the optimization method is either QAT (benchmark) or ISAT (our method), then, we turn the auto optimization off
            self.automatic_optimization = False

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        """Acquire the hyper-parameters from the params"""
        epsilon = self.params["epsilon"] if "epsilon" in self.params else 1
        lam = self.params["lam"] if "lam " in self.params else 1
        opt = self.optimizers()

        if self.params["method"] == "ISTA_LIU" or self.params['method'] == 'QAT':
            torch.set_grad_enabled(False)
            cps, ps = [*self.cmodel.parameters()], [*self.model.parameters()]
            for i, (cp, qp) in enumerate(zip(cps, ps)):
                if (
                        0 <= i < (len(cps))
                ):  # control the layers to be quantized

                    tmp = (cp / self.set_epsilons_[i]).round() * self.set_epsilons_[i]
                    tmp[tmp > self.max_] = self.max_
                    tmp[tmp < self.min_] = self.min_
                    qp.copy_(tmp)
                else:
                    qp.copy_(cp)
            torch.set_grad_enabled(True)

        """Perform the forward pass and backdprop"""
        torch.set_grad_enabled(True)
        self.model.zero_grad()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)

        if self.params["method"] == "QAT" or self.params["method"] == "ISTA_LIU":
            for cp, qp in zip(self.cmodel.parameters(), self.model.parameters()):
                cp.grad = qp.grad  # copy the gradient to the continuous model
        if self.params["method"] == "QAT":
            opt.step_()
        elif self.params["method"] == "ISTA_LIU":
            opt.step_(self.cmodel, x, y)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)

        return {"loss": loss, "acc": acc}

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar(
            "Loss v.s. Step", outputs["loss"].item(), self.global_step
        )
        # self.logger.experiment.add_scalar("Penalty v.s. Step", outputs['penalty'].item(), self.global_step)
        return outputs

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # avg_penalty = torch.stack([x['penalty'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "Loss v.s. Epoch", avg_loss, self.current_epoch
        )
        # self.logger.experiment.add_scalar("Penalty v.s. Epoch", avg_penalty, self.current_epoch)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {"acc": acc}

    def validation_step_end(self, outputs):
        self.logger.experiment.add_scalar(
            "Validation Acc v.s. Step", outputs["acc"], self.global_step
        )
        return outputs

    def validation_epoch_end(self, outputs):
        avg_acc = np.stack([x["acc"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "Validation Acc v.s. Epoch", avg_acc, self.current_epoch
        )
        self.log("acc", avg_acc)
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        _, indices = torch.max(y_hat, dim=1)
        acc = (indices == y).sum().item() / len(y)
        return {"acc": acc}

    def test_epoch_end(self, outputs):
        avg_acc = np.stack([x["acc"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "Test Acc v.s. Epoch", avg_acc, self.current_epoch
        )
        return {"acc": avg_acc}

    def configure_optimizers(self):
        """Setting the optimizer according to hyper-parameter"""
        optimizer = None
        if self.params["method"] == "QAT":
            optimizer = QAT(self.cmodel.parameters(), self.params)
        elif self.params["method"] == "ISTA_LIU":
            optimizer = ISTA_LIU(self.cmodel.parameters(), self.params)
        return optimizer
