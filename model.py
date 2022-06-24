import numpy as np
import pytorch_lightning as pl
import torchvision.models as models
import torch
import timm
import json
from optimizer import ISTA, QAT
from torch.nn import functional as F
from timm.optim.lookahead import Lookahead
from cifar10_models.resnet import resnet18, resnet50
from cifar10_models.resnet20 import resnet20, resnet32, resnet56
from codebase.networks.natnet import NATNet


class Resnet(pl.LightningModule):
    """Resnet model for testing"""

    def __init__(self, params={}):
        """the params records the hyper-parameters of the model and don't confuse with the parameters for the optimizer which stands for the weights of the model."""
        super().__init__()
        self.multiplier = 1
        self.old_parameters = None  # list of paramters from previous iteration
        self.params = params

        # config = json.load(
        #     open("./subnets/cifar100/net-img@224-flops@796-top1@88.3/net.config")
        # )
        # self.model = NATNet.build_from_config(config, pretrained=True)
        # config = json.load(
        #     open("./subnets/cifar100/net-img@224-flops@796-top1@88.3/net.config")
        # )
        # self.cmodel = NATNet.build_from_config(config, pretrained=True)

        # self.model = timm.create_model('efficientnet_l2', pretrained=False)
        # self.cmodel = timm.create_model('resnet50', pretrained=False)
        # self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=False)  # the model that we try to quantize
        # self.model.load_state_dict(torch.load('./cifar100_resnet20-23dac2f1.pt'))
        # self.cmodel = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=False)  # the continuous model

        # self.model = timm.create_model('resnet18', pretrained=True)
        # self.cmodel = timm.create_model('resnet18', pretrained=False)
        # self.model = resnet50()
        # self.cmodel = resnet50()
        # state_dict = torch.load('./cifar10_models/state_dicts/resnet50.pt')
        # self.model.load_state_dict(state_dict)
        # self.cmodel.load_state_dict(state_dict)

        # self.model = resnet32()
        # self.cmodel = resnet32()
        # _state_dict_ = torch.load('./cifar10_models/state_dicts/resnet32-d509ac18.th')['state_dict']
        # state_dict = {}
        # for k, v in _state_dict_.items():
        #     k = k.replace('module.', '')
        #     state_dict[k] = v
        # self.model.load_state_dict(state_dict)
        # self.cmodel.load_state_dict(state_dict)

        self.model = models.resnet50(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cmodel = models.resnet50(num_classes=10)
        self.cmodel.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.names = [p[0] for p in self.model.named_parameters()]
        self.set_epsilons_ = [params['epsilon'] for _ in range(len([*self.model.parameters()]))]
        self.max_ = 1
        self.min_ = -1
        print(self.max_, self.min_)

        if (
                params["method"] == "QAT" or params["method"] == "ISTA"
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

        if self.params["method"] == "QAT":
            torch.set_grad_enabled(False)
            cps, ps = [*self.cmodel.parameters()], [*self.model.parameters()]
            for i, (cp, qp) in enumerate(zip(cps, ps)):
                if (
                        'none' in self.names[i]  # 0 <= i < (len(cps))
                ):  # do not quantize the first and the last layer and copy the continuous gradient to the quantized one and do the quantization accordingly

                    # if self.set_epsilons_[i] is None:
                    #     self.set_epsilons_[i] = epsilon

                    tmp = (cp / self.set_epsilons_[i]).round() * self.set_epsilons_[i]
                    tmp[tmp > 1] = 1
                    tmp[tmp < -1] = -1
                    # qp.copy_((cp / self.set_epsilons_[i]).round() * self.set_epsilons_[i])
                    qp.copy_(tmp)

                    # buf = qp / self.set_epsilons_[i]
                    # buf = torch.unique(buf)
                    # if buf.size()[0] > 2 // epsilon:
                    #     print()
                    #     print('===========')
                    #     print(i, buf.size()[0])
                    #     print('===========')
                else:
                    qp.copy_(cp)
            torch.set_grad_enabled(True)

        # this just quantizes the weights
        # elif self.params["method"] == "ISTA":
        #     cur_params = []
        #     ps = [*self.model.parameters()]
        #
        #     torch.set_grad_enabled(False)
        #     for i, qp in enumerate(ps):
        #         if (
        #                 1 <= i < (len(ps)) - 2
        #         ):  # do not quantize the first and the last layer and copy the continuous gradient to the quantized one and do the quantization accordingly
        #
        #             # if self.set_epsilons_[i] is None:
        #             #     self.set_epsilons_[i] = epsilon
        #
        #             qp.copy_((qp / self.set_epsilons_[i]).round() * self.set_epsilons_[i])
        #             cur_params.append(qp + 0)  # record paramters
        #
        #             # buf = qp
        #             # buf = torch.unique(buf)
        #             # if buf.size()[0] > (2 // epsilon):
        #             #     print()
        #             #     print('===========')
        #             #     print(i, buf.max().item(), self.max_, buf.min(), self.min_)
        #             #     print('===========')
        #
        #     # compare parameters to old parameters and increase multiplier if they are the same
        #     if self.old_parameters:
        #         increase_multiplier = True
        #         for i, (cp, op) in enumerate(zip(cur_params, self.old_parameters)):
        #             if not torch.equal(cp, op):
        #                 increase_multiplier = False
        #                 break
        #         if increase_multiplier == True:
        #             self.multiplier += 1
        #         else:
        #             self.multiplier = 1
        #             # self.multiplier += -(self.multiplier>1) #if the other doesn't work, maybe try this?
        #
        #     # print('=============')
        #     # print(self.multiplier)
        #     # print('*************')
        #     self.old_parameters = cur_params

        """Perform the forward pass and backdprop"""
        torch.set_grad_enabled(True)
        self.model.zero_grad()
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)

        # if self.params["method"] == "QAT" or self.params["method"] == "ISTA":
        if self.params["method"] == "QAT":
            for cp, qp in zip(self.cmodel.parameters(), self.model.parameters()):
                cp.grad = qp.grad  # copy the gradient to the continuous model
        if self.params["method"] == "QAT":
            opt.step_()
        elif self.params["method"] == "ISTA":
            opt.step_(self.model, x, y, multiplier=self.multiplier, set_epsilons_=self.set_epsilons_)
            cur_params = []
            ps = [*self.model.parameters()]
            torch.set_grad_enabled(False)
            for i, qp in enumerate(ps):
                if (
                        'none' in self.names[i]  # 0 <= i < (len(cps))
                ):  # do not quantize the first and the last layer and copy the continuous gradient to the quantized one and do the quantization accordingly

                    # if self.set_epsilons_[i] is None:
                    #     self.set_epsilons_[i] = epsilon

                    tmp = (qp / self.set_epsilons_[i]).round() * self.set_epsilons_[i]
                    tmp[tmp > 1] = 1
                    tmp[tmp < -1] = -1
                    # qp.copy_((qp / self.set_epsilons_[i]).round() * self.set_epsilons_[i])
                    qp.copy_(tmp)
                    cur_params.append(qp + 0)  # record paramters

                    # buf = qp
                    # buf = torch.unique(buf)
                    # if buf.size()[0] > (2 // epsilon):
                    #     print()
                    #     print('===========')
                    #     print(i, buf.max().item(), self.max_, buf.min(), self.min_)
                    #     print('===========')

            # compare parameters to old parameters and increase multiplier if they are the same
            if self.old_parameters:
                increase_multiplier = True
                for i, (cp, op) in enumerate(zip(cur_params, self.old_parameters)):
                    if not torch.equal(cp, op):
                        increase_multiplier = False
                        break
                if increase_multiplier == True:
                    self.multiplier += 1
                else:
                    self.multiplier = 1
                    # self.multiplier += -(self.multiplier>1) #if the other doesn't work, maybe try this?

            # print('=============')
            # print(self.multiplier)
            # print('*************')
            self.old_parameters = cur_params

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
            # optimizer = torch.optim.Adam(self.cmodel.parameters(), lr=self.params["lr"])
        elif self.params["method"] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        elif self.params["method"] == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            optimizer = Lookahead(optimizer, alpha=0.8, k=5)
        elif self.params["method"] == "ISTA":
            # optimizer = ISTA(self.cmodel.parameters(), self.params)
            optimizer = ISTA(self.model.parameters(), self.params)
        return optimizer


class BitM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnetv2_101x3_bitm", pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        """Perform the forward pass and backdprop"""
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

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
            # optimizer = torch.optim.Adam(self.cmodel.parameters(), lr=self.params["lr"])
        elif self.params["method"] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        elif self.params["method"] == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            optimizer = Lookahead(optimizer, alpha=0.8, k=5)
        elif self.params["method"] == "ISTA":
            # optimizer = ISTA(self.cmodel.parameters(), self.params)
            optimizer = ISTA(self.model.parameters(), self.params)
        return optimizer
